import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import answer
import rag_index


APP = Path(__file__).resolve().parents[1] / 'app.py'


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index_dir = Path(self.tmp.name)
        for name in ('info.json', 'meta.jsonl', 'vectors.npy', 'bm25.pkl'):
            (self.index_dir / name).write_text('test fixture')
        env = patch.dict(os.environ, {'POLITICS_INDEX_DIR': self.tmp.name,
                                     'GEMINI_API_KEY': 'test-key', 'GOOGLE_API_KEY': ''})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(st.cache_resource.clear)
        self.fake_index = SimpleNamespace(meta=[{'layer': 'said'}, {'layer': 'motion'}])
        loader = patch.object(rag_index, 'Index', return_value=self.fake_index)
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        responder = patch.object(answer, 'answer_with_tools')
        self.responder = responder.start()
        self.addCleanup(responder.stop)
        self.responder.return_value = ('Ett förslag [1].', [
            (1, {'layer': 'said', 'url': 'https://example.org/politik'}),
            (0.8, {'layer': 'said', 'url': 'https://example.org/politik'})],
            {'hits': 2, 'model': 'test-model'}, [])

    def app(self):
        return AppTest.from_file(str(APP), default_timeout=15).run()

    def test_missing_key_prevents_calls(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': '', 'GOOGLE_API_KEY': ''}):
            app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertIn('API-nyckel saknas', app.warning[0].value)
        self.responder.assert_not_called()

    def test_incomplete_index_prevents_calls(self):
        (self.index_dir / 'vectors.npy').unlink()
        app = self.app()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertIn('vectors.npy', app.error[0].value)
        self.responder.assert_not_called()

    def test_submit_render_and_rerun_do_not_repeat_request(self):
        app = self.app()
        app.chat_input[0].set_value('Vad vill S?').run()
        self.assertFalse(app.exception)
        self.responder.assert_called_once_with(self.fake_index, 'Vad vill S?', model=answer.MODEL)
        self.assertTrue(any('Ett förslag [1].' in m.value for m in app.markdown))
        result = app.session_state['result']
        self.assertEqual(result['sources'], ['[1, 2] https://example.org/politik'])
        self.assertEqual(len(result['links']), 1)
        app.run()
        self.responder.assert_called_once()
        app.chat_input[0].set_value('Vad vill V?').run()
        self.assertEqual(self.responder.call_count, 2)
        self.assertEqual(self.loader.call_count, 1)

    def test_api_failure_is_safe_and_retry_is_possible(self):
        self.responder.side_effect = RuntimeError('401 UNAUTHENTICATED secret-test-value')
        app = self.app()
        app.chat_input[0].set_value('En fråga').run()
        self.assertFalse(app.exception)
        self.assertIn('godkände inte åtkomsten', app.error[0].value)
        self.assertNotIn('secret-test-value', app.error[0].value)
        self.assertNotIn('result', app.session_state)
        self.responder.side_effect = None
        app.chat_input[0].set_value('Försök igen').run()
        self.assertIn('result', app.session_state)

    def test_backend_system_exit_is_handled(self):
        self.responder.side_effect = SystemExit('daily quota is used up')
        app = self.app()
        app.chat_input[0].set_value('En fråga').run()
        self.assertFalse(app.exception)
        self.assertIn('Gränsen för Gemini-anrop', app.error[0].value)

    def test_whitespace_key_is_rejected(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'bad key'}):
            app = self.app()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertIn('blanktecken', app.warning[0].value)
        self.responder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
