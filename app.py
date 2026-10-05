"""Lokalt webbgränssnitt. Starta från projektmappen med streamlit run app.py."""

import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

import streamlit as st

import answer as bot
import rag_index


ROOT = Path(__file__).resolve().parent
INDEX_FILES = ('info.json', 'meta.jsonl', 'vectors.npy', 'bm25.pkl')
LAYER_NAMES = {'said': 'partisidor', 'did': 'riksdagsmaterial',
               'motion': 'motioner', 'program': 'partiprogram'}


@st.cache_resource(show_spinner=False)
def load_index(path, fingerprint):
    # Låset skyddar den delade sökmodellen om flera lokala flikar används.
    return rag_index.Index(path), threading.Lock()


def error_message(error):
    # Visa inte råa API-fel: de kan innehålla tekniska detaljer eller nycklar.
    text = str(error).lower()
    if any(word in text for word in ('401', '403', 'api_key_invalid', 'unauthenticated')):
        return 'Gemini godkände inte åtkomsten. Kontrollera API-nyckeln och dess behörigheter.'
    if any(word in text for word in ('429', 'quota', 'resource_exhausted')):
        return 'Gränsen för Gemini-anrop har nåtts. Vänta och försök igen senare.'
    if any(word in text for word in ('404', 'not_found', 'not available')):
        return 'Den valda modellen är inte tillgänglig. Kontrollera standardmodellen i answer.py.'
    if isinstance(error, (FileNotFoundError, OSError)):
        return 'En fil eller anslutning kunde inte öppnas. Kontrollera indexfilerna och nätverket.'
    return 'Svaret kunde inte hämtas. Kontrollera nätverket, API-åtkomsten och indexet och försök igen.'


def render_result(result):
    with st.chat_message('user'):
        st.write(result['question'])
    with st.chat_message('assistant'):
        st.markdown(result['answer'])
        with st.expander('Källor och hämtat underlag', expanded=True):
            st.caption('Listan visar hämtade källor, även sådana som inte används i svaret. '
                       'Flera nummer kan hänvisa till samma dokument.')
            if not result['sources']:
                st.write('Inga källor hämtades för den här frågan.')
            for source in result['sources']:
                st.write(source)
            for label, url in result['links']:
                st.link_button(label, url)
        with st.expander('Om denna sökning'):
            st.write('Material: ' + ', '.join(result['layers']))
            st.write(f"Hämtade textavsnitt: {result['info'].get('hits', 0)}")
            st.write(f"Modell: {result['info'].get('model', '')}")


def main():
    st.set_page_config(page_title='Svensk politik – fråga källorna', page_icon='💬')
    st.title('Svensk politik')
    st.write('Utforska vad partierna säger, föreslår och gör i riksdagen.')
    st.caption('Svar baseras på sparade källor och kan vara ofullständiga. '
               'Varje fråga behandlas separat – skriv ut parti och ämne även i följdfrågor.')

    index_path = Path(os.environ.get('POLITICS_INDEX_DIR', ROOT / 'out/index'))
    model = bot.MODEL

    ready = True
    if Path.cwd().resolve() != ROOT:
        st.error('Starta appen från projektets huvudmapp så att boten hittar sina datafiler.')
        ready = False
    missing = [name for name in INDEX_FILES if not (index_path / name).is_file()]
    if missing:
        st.error('Det valda indexet är ofullständigt. Saknade filer: ' + ', '.join(missing))
        ready = False
    key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not key:
        st.warning('API-nyckel saknas. Ange GEMINI_API_KEY i terminalen och starta om appen.')
        ready = False
    elif any(c.isspace() for c in key):
        st.warning('API-nyckeln innehåller blanktecken. Kopiera in nyckeln igen och starta om appen.')
        ready = False

    with st.expander('Exempel på frågor'):
        for example in ('Vad tycker Vänsterpartiet om kärnkraft?',
                        'Vilka motioner har S lagt om gifttunnor i Sundsvallsbukten?',
                        'Hur röstade S om kärnkraft?'):
            st.write(example)

    question = st.chat_input('Skriv din fråga om svensk politik', disabled=not ready)
    if question and question.strip():
        st.session_state.pop('result', None)
        try:
            with st.spinner('Läser underlaget, söker och formulerar svar…'):
                fingerprint = tuple((name, (index_path / name).stat().st_mtime_ns,
                                     (index_path / name).stat().st_size) for name in INDEX_FILES)
                idx, lock = load_index(str(index_path), fingerprint)
                with lock:
                    response, hits, info, _ = bot.answer_with_tools(idx, question.strip(), model=model)
                links, seen = [], set()
                for number, (_, meta) in enumerate(hits, 1):
                    url = meta.get('url', '')
                    parsed = urlsplit(url)
                    if parsed.scheme in ('https', 'http') and parsed.netloc and url not in seen:
                        links.append((f'Öppna originalkälla [{number}]', url))
                        seen.add(url)
                st.session_state.result = {
                    'question': question.strip(), 'answer': response,
                    'sources': bot.source_list(hits), 'links': links, 'info': info,
                    'layers': [LAYER_NAMES.get(layer, layer) for layer in
                               sorted({m['layer'] for m in idx.meta})],
                }
        except (Exception, SystemExit) as error:
            st.error(error_message(error))
    if 'result' in st.session_state:
        render_result(st.session_state.result)


if __name__ == '__main__':
    main()
