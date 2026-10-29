from __future__ import annotations

import unittest

from scripts.build_quiz_index import _enforce_cost_approval, normalize_quiz_data


class QuizIndexNormalizationTest(unittest.TestCase):
    def test_merges_overrides_and_keeps_quiz_and_diagnostic_answers(self) -> None:
        quizzes = [
            {
                "id": "Q001",
                "category": "주식",
                "difficulty": "초급",
                "type": "4지선다",
                "question": "원본 문제",
                "choices": ["A", "B"],
                "answer": "B",
                "explanation": "원본 해설",
                "key_concept": "지분",
                "learning_goal": "주식 이해",
                "source_name": "공식 출처",
                "source_url": "https://example.test",
                "verification_status": "VERIFIED",
            }
        ]
        diagnostics = [
            {
                "diagnostic_quiz_id": "S1",
                "question_no": 1,
                "category": "주식",
                "difficulty": "초급",
                "question_type": "객관식",
                "question_text": "진단 문제",
                "choice_1": "오답",
                "choice_2": "정답",
                "answer": 2,
                "explanation": None,
                "weakness_tag": "주식 개념",
            },
            {
                "diagnostic_quiz_id": "S1",
                "question_no": 2,
                "question_text": "정답이 없는 진단 문제",
                "choice_1": "보기",
                "answer": None,
                "explanation": None,
            },
        ]

        rows = normalize_quiz_data(
            quizzes,
            {"Q001": {"question": "수정 문제", "explanation": "수정 해설"}},
            diagnostics,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["question"], "수정 문제")
        self.assertEqual(rows[0]["answer"], "B")
        self.assertEqual(rows[0]["explanation"], "수정 해설")
        self.assertIn("정답: B", rows[0]["embedding_text"])
        self.assertIn("해설: 수정 해설", rows[0]["embedding_text"])
        self.assertEqual(rows[1]["answer"], "2. 정답")
        self.assertIn("정답: 2. 정답", rows[1]["embedding_text"])
        self.assertIn("해설: 미제공", rows[1]["embedding_text"])
        self.assertIn("정답: 미제공", rows[2]["embedding_text"])
        self.assertEqual(len({row["chunk_id"] for row in rows}), 3)

    def test_rejects_overrides_for_unknown_quizzes(self) -> None:
        with self.assertRaisesRegex(ValueError, "원본에 없는 ID"):
            normalize_quiz_data([], {"unknown": {"question": "x"}}, [])

    def test_requires_cost_approval_before_embedding(self) -> None:
        with self.assertRaisesRegex(SystemExit, "임베딩 API는 호출하지 않았습니다"):
            _enforce_cost_approval(0.01, None)

    def test_rejects_cost_above_approved_limit(self) -> None:
        with self.assertRaisesRegex(SystemExit, "초과해 임베딩 API를 호출하지 않았습니다"):
            _enforce_cost_approval(0.01, 0)


if __name__ == "__main__":
    unittest.main()
