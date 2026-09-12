import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import answer
import rag_index
from scrapers.motions import select, extract_text, collect


def person(pid, party="V", role="undertecknare"):
    return {"intressent_id": pid, "namn": f"Person {pid}", "partibet": party, "roll": role}


def document(people, title="Ett förslag"):
    return {"titel": title, "dokintressent": {"intressent": people}}


class MotionTests(unittest.TestCase):
    def test_single_and_two(self):
        self.assertEqual(select(document(person("1")))[1], "en_undertecknare")
        self.assertEqual(select(document([person("1"), person("2")]))[1], "inkludera")

    def test_duplicate_and_other_roles_do_not_count(self):
        self.assertEqual(select(document([person("1"), person("1"), person("2", role="annan")]))[1],
                         "en_undertecknare")

    def test_missing_and_withdrawn(self):
        self.assertEqual(select({"titel": "Motionen utgår"})[1], "utgar")
        self.assertEqual(select({"titel": "Test"})[1], "saknar_undertecknare")
        self.assertEqual(select(document([person("1"), person("")]))[1], "ofullstandiga_undertecknare")

    def test_multiple_parties_and_conflicts(self):
        authors, status = select(document([person("1", "V"), person("2", "MP")]))
        self.assertEqual(status, "inkludera")
        self.assertEqual({p["partibet"] for p in authors}, {"V", "MP"})
        self.assertEqual(select(document([person("1", "V"), person("1", "MP")]))[1],
                         "ofullstandiga_undertecknare")

    def test_html_keeps_words_and_removes_style(self):
        raw = '<style>bad css</style><p>Vi <b>föreslår</b> solkraft.</p><p>Av\xadveckling &amp; stöd.</p>'
        self.assertEqual(extract_text(raw), 'Vi föreslår solkraft.\n\nAvveckling & stöd.')

    def test_pagination_and_limit(self):
        def page(n):
            return json.dumps({"dokumentlista": {"@traffar": "2", "@nasta_sida": "next" if n == 1 else "",
                "dokument": {"dok_id": f"A{n}", "rm": "2023/24", "doktyp": "mot"}}})
        with patch('scrapers.motions.get_cached', side_effect=[page(1), page(2)]):
            self.assertEqual(len(collect(['2023/24'], Path('/unused'))), 2)
        with patch('scrapers.motions.get_cached', return_value=page(1)) as get:
            self.assertEqual(len(collect(['2023/24'], Path('/unused'), limit=1)), 1)
            self.assertEqual(get.call_count, 1)

    def test_duplicate_pages_are_not_silently_accepted(self):
        page = json.dumps({"dokumentlista": {"@traffar": "2", "@nasta_sida": "next",
            "dokument": {"dok_id": "A1", "rm": "2023/24", "doktyp": "mot"}}})
        with patch('scrapers.motions.get_cached', return_value=page):
            with self.assertRaisesRegex(ValueError, 'Upprepat dokument'):
                collect(['2023/24'], Path('/unused'))

    def test_routing_and_party_names(self):
        self.assertEqual(answer.valj_lager('Vilka motioner har V lämnat?'), 'motion')
        self.assertEqual(answer.valj_lager('Vilka motioner lade flera partier om tillgänglighet i riksdagens lokaler?'), 'motion')
        self.assertEqual(answer.valj_lager('Vad tycker Vänsterpartiet om kärnkraft?'), 'said')
        self.assertIsNone(answer.valj_lager('Vilka motioner lade V och hur röstade de?'))
        self.assertEqual(answer.parti_i_fragan('Vänsterpartiets motioner'), ['V'])

    def test_citations_keep_every_passage_number(self):
        hits = [(1, {"url": "https://example.org/a"}), (0.5, {"url": "https://example.org/a"})]
        self.assertEqual(answer.kallista(hits), ['[1] https://example.org/a', '[2] https://example.org/a'])

    def test_only_cited_sources_keep_original_numbers(self):
        hits = [(1, {"url": f"https://example.org/{n}"}) for n in range(1, 7)]
        hits[4] = hits[0]  # Olika avsnitt i samma dokument får inte tappa sitt nummer.
        self.assertEqual(answer.kallista(hits, "Förslag [1, 5]. Bakgrund [6]. [1]"),
                         ['[1] https://example.org/1', '[5] https://example.org/1',
                          '[6] https://example.org/6'])

    def test_citation_ranges_and_non_numeric_references(self):
        hits = [(1, {"url": f"https://example.org/{n}"}) for n in range(1, 7)]
        self.assertEqual(len(answer.kallista(hits, "[1–3, 5-6]")), 5)
        self.assertEqual(answer.kallista(hits, "[AU10 2022/23 punkt 1] [99] [3-1]"), [])
        self.assertEqual(answer.kallista(hits, "Inget underlag."), [])
        self.assertEqual(len(answer.kallista(hits, "[5–999999999]")), 2)

    def test_printed_sources_exclude_uncited_hits(self):
        from contextlib import redirect_stdout
        from io import StringIO
        hits = [(1, {"url": "https://example.org/cited"}),
                (0.5, {"url": "https://example.org/unused"})]
        info = {"partier": ['V'], "lager": 'motion', "träffar": 2,
                "röstrader": 0, "kontexttecken": 100, "avrådan": False}
        output = StringIO()
        with redirect_stdout(output):
            answer.skriv("Förslag [1].", hits, info, [], False)
        self.assertIn('[1] https://example.org/cited', output.getvalue())
        self.assertNotIn('https://example.org/unused', output.getvalue())

    def test_named_vote_citations_match_retrieved_cases(self):
        meta = {"layer": "did", "beteckning": "NU20", "rm": "2024/25", "punkt": "4"}
        other = {"layer": "did", "beteckning": "NU5", "rm": "2023/24", "punkt": 2}
        hits = [(1, meta), (0.9, meta), (0.8, other)]
        result = answer.kallista(hits, "[NU20 2024/25 punkt 4, NU5 2023/24 punkt 2]. "
                                 "[NU20 2024/25 punkt 4] [NU99 2024/25 punkt 9]")
        self.assertEqual(result, [
            '[NU20 2024/25 punkt 4] Betänkande 2024/25:NU20, punkt 4',
            '[NU5 2023/24 punkt 2] Betänkande 2023/24:NU5, punkt 2'])
        self.assertEqual(len(answer.kallista(hits, '[1] [NU20 2024/25 punkt 4]')), 2)
        self.assertEqual(answer.kallista([(1, dict(meta, layer='motion'))],
                                        '[NU20 2024/25 punkt 4]'), [])

    def test_mixed_citations_keep_numeric_and_named_sources(self):
        hits = [(1, {"url": f"https://example.org/{i}"}) for i in range(1, 11)]
        hits[0] = (1, {"layer": "did", "beteckning": "NU5", "rm": "2023/24", "punkt": "2"})
        hits[4] = (1, {"layer": "did", "beteckning": "NU5", "rm": "2023/24", "punkt": "4"})
        hits[5] = (1, {"layer": "did", "beteckning": "NU20", "rm": "2024/25", "punkt": "5"})
        result = answer.kallista(hits, '[NU5 2023/24 punkt 2, 8] '
                                '[NU5 2023/24 punkt 4, 2, 3] [NU20 2024/25 punkt 5, 7]')
        self.assertEqual(result[:4], [f'[{i}] https://example.org/{i}' for i in [2, 3, 7, 8]])
        self.assertEqual(len(result), 7)
        result = answer.kallista(hits, '[8, NU5 2023/24 punkt 2, 9–10]')
        self.assertEqual(result[:3], [f'[{i}] https://example.org/{i}' for i in [8, 9, 10]])
        self.assertEqual(len(result), 4)
        self.assertEqual(answer.kallista(hits, '[NU5 2023/24 punkt 2]'),
                         ['[NU5 2023/24 punkt 2] Betänkande 2023/24:NU5, punkt 2'])

    def test_missing_votes_do_not_assert_acclamation_or_non_participation(self):
        import party_positions
        row = {"beteckning": "NU20", "rm": "2024/25", "punkt": "5",
               "status": "no_vote", "voteringar": [],
               "reservationer": [{"number": 10, "partier": ['S']}]}
        result = party_positions.describe(row, 'S')
        self.assertIn('Voteringsuppgift saknas', result['text'])
        self.assertIn('Det går därför inte att avgöra', result['text'])
        self.assertNotIn('ärendet avgjordes med acklamation', result['text'])
        self.assertEqual(result['egna_reservationer'], [10])
        row.update(status='voted', voteringar=[{"partier": {}}])
        result = party_positions.describe(row, 'S')
        self.assertIn('Röstuppgift för Socialdemokraterna saknas', result['text'])
        self.assertNotIn('Socialdemokraterna deltog inte', result['text'])

    def test_motive_vote_stances_and_split_counts(self):
        import party_positions
        cases = [
            ('Nej', 0, 93, 0, 14, 'S röstade för en motivreservation'),
            ('Ja', 93, 0, 0, 14, 'S röstade med utskottets förslag mot'),
            ('Avstår', 0, 0, 93, 14, 'S avstod i omröstningen'),
            ('Frånvarande', 0, 0, 0, 107, 'S var frånvarande'),
            ('Avstår', 3, 0, 90, 14, 'Ledamöterna från S röstade olika'),
            ('Avstår', 2, 0, 91, 14, 'Ledamöterna från S röstade olika'),
        ]
        for stance, ja, nej, avstod, franvarande, expected in cases:
            with self.subTest(stance=stance, ja=ja):
                info = dict(stance=stance, ja=ja, nej=nej, avstod=avstod, franvarande=franvarande)
                row = dict(beteckning='NU5', rm='2023/24', punkt='2', status='voted',
                           reservationer=[], voteringar=[dict(avser='motivfrågan',
                           datum='2023-11-29', resolution='motivfraga', partier={'S': info})])
                text = party_positions.describe(row, 'S')['text']
                self.assertIn(expected, text)
                self.assertIn(f'{ja} ja, {nej} nej, {avstod} avstod och {franvarande} frånvarande', text)
                self.assertIn('2023-11-29', text)
                self.assertIn('motiveringen, inte sakfrågan', text)
                if stance in ('Avstår', 'Frånvarande'):
                    self.assertNotIn('S röstade mot', text)
                    self.assertNotIn('S röstade för', text)

    def test_motion_passages_keep_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'chunks').write_text('')
            row = {"chunk_id": "motion:A", "rm": "2023/24", "beteckning": "1", "layer": "motion",
                   "heading": "Kärnkraft", "datum": "2023-10-01", "parti": "V;MP",
                   "undertecknare": [person('1'), person('2', 'MP')], "text": "Lång text. " * 200}
            (p/'motions').write_text(json.dumps(row))
            with patch.object(rag_index, 'SAID_GLOB', str(p/'missing*')), patch.object(rag_index, 'DID_CHUNKS', str(p/'chunks')):
                passages = rag_index.load_records(p/'motions')
            self.assertGreater(len(passages), 1)
            self.assertTrue(all(m['layer']=='motion' and m['parti']=='V;MP' for m in passages))
            self.assertIn('Inte ett beslut', passages[0]['kontext'])
            encoded = rag_index.passage_text(passages[0])
            self.assertIn('kärnkraft'.capitalize(), encoded)
            self.assertNotIn('Person 1', encoded)
            self.assertIn('Person 1', passages[0]['kontext'])

    def test_filters_before_candidate_cutoff(self):
        # Den relevanta motionen får inte försvinna bakom många andra källtyper.
        idx = rag_index.Index.__new__(rag_index.Index)
        idx.meta = [{"layer": "said", "parti": "V", "text": "kärnkraft"} for _ in range(5)]
        idx.meta += [{"layer": "motion", "parti": "V", "rm": "2023/24", "text": "kärnkraft"}]
        idx.bm = rag_index.BM25([rag_index.tokenize(m['text']) for m in idx.meta])
        idx.vek = np.ones((6, 2), dtype='float32')
        idx._encode_query = lambda q: np.ones(2, dtype='float32')
        for method in ['bm25', 'vektor', 'hybrid']:
            hits = idx.search('kärnkraft', layer='motion', parti=['V'], rm='2023/24', kandidater=2, metod=method)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0][1]['layer'], 'motion')

    def test_motion_does_not_become_vote_lookup(self):
        with patch.object(answer.party_positions, 'load_positions', return_value={}) as load:
            answer.rostfakta([(1, {"layer": "motion", "rm": "2023/24", "beteckning": "1"})], ['V'])
            # Inget krav på beslutspunkt i motionen och inga påhittade röstrader.
            self.assertEqual(answer.rostfakta([(1, {"layer": "motion"})], ['V']), [])

    def test_dry_run_routes_year_without_calling_gemini(self):
        class FakeIndex:
            def search(self, question, **kwargs):
                self.kwargs = kwargs
                return []
        idx = FakeIndex()
        with patch.object(answer, 'fraga_gemini', side_effect=AssertionError('API får inte anropas')):
            result = answer.svara(idx, 'Vänsterpartiets motioner 2023/24', torrkor=True)
        self.assertEqual(idx.kwargs['rm'], '2023/24')
        self.assertEqual(idx.kwargs['layer'], 'motion')
        self.assertIsNone(result[0])

    def test_reuse_preserves_base_vectors_and_rejects_changed_text(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'base'
            new = Path(directory)/'new'
            base.mkdir()
            record = {"id": "old", "layer": "said", "parti": "V", "kontext": "V", "text": "kärnkraft"}
            motion = {"id": "new", "layer": "motion", "parti": "V", "kontext": "motion", "text": "solkraft",
                      "rm": "2023/24", "beteckning": "1", "heading": "Solkraft"}
            (base/'meta.jsonl').write_text(json.dumps(record)+'\n')
            (base/'info.json').write_text(json.dumps({"model": "test-model", "dim": 2, "n": 1}))
            np.save(base/'vectors.npy', np.array([[1., 0.]], dtype='float32'))
            rag_index.BM25([['kärnkraft']]).save(base/'bm25.pkl')
            with patch.object(rag_index, 'load_records', return_value=[record, motion]), \
                 patch.object(rag_index, 'encode', return_value=np.array([[0., 1.]], dtype='float32')) as encode:
                rag_index.build(out_dir=str(new), base_index=str(base))
                self.assertEqual(len(encode.call_args.args[0]), 1)
            np.testing.assert_array_equal(np.load(base/'vectors.npy'), [[1., 0.]])
            np.testing.assert_array_equal(np.load(new/'vectors.npy'), [[1., 0.], [0., 1.]])
            with patch.object(rag_index, 'load_records', return_value=[{**record, 'text': 'ändrat'}]):
                with self.assertRaises(ValueError):
                    rag_index.build(out_dir=str(new), base_index=str(base))
                with self.assertRaises(ValueError):
                    rag_index.build(out_dir=str(base), bara_bm25=True)
            self.assertEqual(json.loads((base/'meta.jsonl').read_text()), record)


if __name__ == '__main__':
    unittest.main()
