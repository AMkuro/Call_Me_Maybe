"""Handle one prompt: name -> skeleton -> values -> check."""

from .model_port import Model
from .name_trie import NameTrie
from .schema import FunctionCall, FunctionDef


def select_name(
    user_prompt: str,
    functions: list[FunctionDef],
    model: Model,
    trie: NameTrie,
) -> FunctionDef:
    """Let the model choose a function, limited by the name tree."""
    raise NotImplementedError


def generate_parameters(
    user_prompt: str,
    fn: FunctionDef,
    model: Model,
    vocab: dict[int, str],
) -> str:
    """Generate the parameters, forcing the skeleton.

    Returns the full JSON text of the record.

    Judgment: this function builds the head ``{"prompt": ..., "name": "...``
    itself and adds the rest from the skeleton and the generated values.
    Open: should the skeleton include the head?
    """
    raise NotImplementedError


def validate_call(text: str, fn: FunctionDef) -> FunctionCall:
    """Parse ``text`` as JSON and check it with the dynamic model."""
    raise NotImplementedError


def process_prompt(
    user_prompt: str,
    functions: list[FunctionDef],
    model: Model,
    trie: NameTrie,
    vocab: dict[int, str],
) -> FunctionCall:
    """Run the whole flow for one prompt. Raises PromptError on failure."""
    raise NotImplementedError
