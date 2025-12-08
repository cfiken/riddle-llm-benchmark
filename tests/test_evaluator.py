import pytest
from pathlib import Path
from riddle_benchmark.evaluation.evaluator import Evaluator
from riddle_benchmark.dataset.schema import Riddle

@pytest.fixture
def mock_riddle():
    return Riddle(
        id="1",
        image_path=Path("img.png"),
        question="q",
        acceptable_answers=["りんご", "Apple", "◯"]
    )

@pytest.mark.parametrize("input_text, expected", [
    ("りんご", "りんご"),
    ("リンゴ", "リンゴ"),  # NFKC doesn't convert Katakana to Hiragana
    ("Ａｐｐｌｅ", "apple"), # Fullwidth to halfwidth + lower
    ("  apple  ", "apple"), # Whitespace removal
    ("ap\nple", "apple"),   # Newline removal
    ("◯", "◯"),           # Symbol preservation
    ("□", "□"),           # Symbol preservation
    ("１２３", "123"),    # Fullwidth numbers to halfwidth
])
def test_normalize(input_text, expected):
    assert Evaluator.normalize(input_text) == expected

def test_evaluate_correct(mock_riddle):
    assert Evaluator.evaluate("りんご", mock_riddle)
    assert Evaluator.evaluate("apple", mock_riddle)
    assert Evaluator.evaluate("Ａｐｐｌｅ", mock_riddle) # Normalized match
    assert Evaluator.evaluate("  りんご  ", mock_riddle)
    assert Evaluator.evaluate("◯", mock_riddle) # Symbol match

def test_evaluate_incorrect(mock_riddle):
    assert not Evaluator.evaluate("みかん", mock_riddle)
    assert not Evaluator.evaluate("banana", mock_riddle)
    assert not Evaluator.evaluate("りんごです", mock_riddle) # Exact match requirement (normalized)
    assert not Evaluator.evaluate("✕", mock_riddle) # Wrong symbol
