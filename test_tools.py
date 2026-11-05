"""Offline tests for the knowledge-base tool (no Ollama needed). Run: python -m unittest -v"""
import unittest

from tools import execute_tool, load_knowledge_base, search_knowledge_base


class SearchTests(unittest.TestCase):
    def test_kb_has_at_least_10_entries(self):
        self.assertGreaterEqual(len(load_knowledge_base()), 10)

    def test_ram_query(self):
        r = search_knowledge_base("my computer is slow and RAM usage is 95%")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["results"][0]["id"], "KB-001")

    def test_dns_query(self):
        r = search_knowledge_base("dns server not responding")
        self.assertEqual(r["results"][0]["id"], "KB-005")

    def test_printer_query(self):
        self.assertEqual(search_knowledge_base("printer not responding")["results"][0]["id"], "KB-007")

    def test_wifi_variants(self):
        self.assertEqual(search_knowledge_base("Wi-Fi keeps disconnecting")["results"][0]["id"], "KB-004")

    def test_not_whole_database(self):
        r = search_knowledge_base("printer offline")
        self.assertLess(len(r["results"]), len(load_knowledge_base()))

    def test_unrelated_query_gives_no_results(self):
        self.assertEqual(search_knowledge_base("how to bake a chocolate cake")["status"], "no_results")

    def test_max_results_respected(self):
        self.assertLessEqual(len(search_knowledge_base("slow", 1)["results"]), 1)


class ValidationTests(unittest.TestCase):
    def test_missing_query(self):
        self.assertEqual(execute_tool("search_knowledge_base", {})["status"], "error")

    def test_empty_query(self):
        self.assertEqual(execute_tool("search_knowledge_base", {"query": "  "})["status"], "error")

    def test_bad_type(self):
        self.assertEqual(execute_tool("search_knowledge_base", {"query": 5})["status"], "error")

    def test_string_arguments(self):
        self.assertEqual(execute_tool("search_knowledge_base", '{"query": "high cpu"}')["status"], "ok")

    def test_invalid_json_string(self):
        self.assertEqual(execute_tool("search_knowledge_base", "{not json")["status"], "error")

    def test_unknown_tool(self):
        self.assertEqual(execute_tool("run_command", {"cmd": "dir"})["status"], "error")


if __name__ == "__main__":
    unittest.main()
