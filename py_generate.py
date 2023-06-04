import openai
import random

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

from typing import Optional, List, Union
import ast
import re
from dataclasses import dataclass


PY_REFLEXION_CHAT_INSTRUCTION = """
You are PythonGPT. You will be given your past function implementation, 
a series of unit tests, and a hint to change the implementation appropriately. 
Apply the changes below by writing the body of this function only. 
You should fill in the following text of the missing function body. 
For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."""
PY_REFLEXION_CHAT_INSTRUCTION_V2 = """
You are PythonGPT. You will be given your previous implementation of a function, 
a series of unit tests results, and your self-reflection on your previous implementation. 
Apply the necessary changes below by responding only with the improved body of the function. 
Do not include the signature in your response. 
The first line of your response should have 4 spaces of indentation so that it fits syntactically with the user provided signature. 
You will be given a few examples by the user."""


PY_REFLEXION_FEW_SHOT = '''Example 1:
[previous impl]:
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert fullJustify([], 10) == [] # output: ['          ']
assert fullJustify([], 0) == [] # output: ['']

[reflection on previous impl]:
The implementation failed the test cases where the input list of words is empty. The issue arises because the code does not handle the case where there are no words to process. As a result, it still appends a line with spaces to the result list, even when there are no words. To fix this issue, we should add a condition at the beginning of the function to check if the input list is empty, and return an empty list if it is. This will ensure that the function returns the correct output for empty input lists.

[improved impl]:
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    if not words:
        return []

    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
END EXAMPLES
'''

PY_SIMPLE_CHAT_INSTRUCTION = """
You are PythonGPT, an AI that only responds with only python code. 
You will be given a function signature and its docstring by the user. 
Respond only in code with a correct, efficient implementation of the function. 
Do not include provided the docstring in your response."""  # The first line of your response should have 4 spaces of indentation so that it fits syntactically with the user provided signature.


PY_SELF_REFLECTION_CHAT_INSTRUCTION = """
You are PythonGPT. You will be given a function implementation and a series of unit tests. 
Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. 
You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."""

PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = """
You are PythonGPT. You will be given a function implementation and a series of unit test results. 
Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. 
You will need this as guidance when you try again later. 
Only provide the few sentence description in your answer, not the implementation. 
You will be given a few examples by the user."""
PY_SELF_REFLECTION_FEW_SHOT = """Example 1:
[function impl]:
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []  
Tests failing:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]
[self-reflection]:
The implementation failed the where no subarray fulfills the condition. The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition.

Example 2:
[function impl]:
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []
Tests failing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3] # output: list index out of range
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5] # output: list index out of range
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: list index out of range
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3] # output: list index out of range
[self-reflection]:
The implementation failed 4 out of the 7 test cases due to an IndexError. The issue stems from the while loop while current_sum + nums[right] <= target:, which directly accesses nums[right] without checking if right is within the bounds of the list. This results in a runtime error when right goes beyond the list length. To overcome this error, we need to add a bounds check for the right variable in the mentioned while loop. We can modify the loop condition to while right < len(nums) and current_sum + nums[right] <= target:. This change will ensure that we only access elements within the bounds of the list, thus avoiding the IndexError.
END OF EXAMPLES
"""

PY_TEST_GENERATION_FEW_SHOT = """
Example:
=====func signature start=====
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
=====func signature end=====
test numbers: 7
unit tests:
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False

=====func signature start=====
{func_sig}
=====func signature end=====
test numbers: {num_tests}
unit tests:"""



PY_TEST_GENERATION_CHAT_INSTRUCTION = """
You are CodexGPT, an AI coding assistant that can write unique, diverse, 
and intuitive unit tests fo
"""


@dataclass
class GenerateResult:
    """Generate Result that contains system message, user message, and generated result"""
    
    system_message: str
    user_message: str
    generated_string: str
    processed_result: any
    



class PyGenerator:
    def __init__(self, model: str = "gpt-4"):
        if model not in ["gpt-4", "gpt-3.5-turbo"]:
            raise ValueError(
                f"Invalid model: given `{model}` but expected one of `gpt-4` or `gpt-3.5-turbo`"
            )
        self.model = model

    def self_reflection(self, func: str, feedback: str) -> str:
        system_message = PY_SELF_REFLECTION_CHAT_INSTRUCTION
        message = f"{PY_SELF_REFLECTION_FEW_SHOT}\n\n[function impl]:\n{func}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:"
        reflection = gpt_chat(
            self.model,
            system_message,
            message,
        )
        # print(f"Self reflection output: {reflection}")
        return GenerateResult(
            system_message, message, reflection, reflection
        )

    def func_impl(
        self,
        func_sig: str,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        if strategy not in ["simple", "reflexion"]:
            raise ValueError(
                f"Invalid strategy: given `{strategy}` but expected one of `simple` or `reflexion`"
            )
            
        if strategy == "reflexion":
            system_message = PY_REFLEXION_CHAT_INSTRUCTION
            message = f"{PY_REFLEXION_FEW_SHOT}\n[previous impl]:\n{prev_func_impl}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:\n{func_sig}"
        else:
            system_message = PY_SIMPLE_CHAT_INSTRUCTION
            message = func_sig
        # print("----------------------- SYSTEM MESSAGE -----------------------")
        # print(system_message)
        # print(" ----------------------- USER MESSAGE -----------------------")
        # print(message, flush=True)
        func_bodies = gpt_chat(
            self.model,
            system_message,
            message,
            num_comps=num_comps,
            temperature=temperature,
        )       

        assert isinstance(func_bodies, str)
        # print("--------------------- GENERATED FUNC BODY ---------------------")
        # print(func_sig + py_fix_indentation(func_bodies))
        return GenerateResult(
            system_message, 
            message, 
            func_bodies,
            func_sig +"\n" +  py_fix_indentation(func_bodies)
        )




    def internal_tests(self, func_sig: str, max_num_tests: int = 5) -> List[str]:
        def parse_tests(tests: str) -> List[str]:
            return [test.strip() for test in tests.splitlines() if "assert" in test]

        message = (
            PY_TEST_GENERATION_FEW_SHOT.format(
                func_sig=func_sig, num_tests=max_num_tests + 1
            )
        )
        output = gpt_chat(
            self.model, PY_TEST_GENERATION_CHAT_INSTRUCTION, message, max_tokens=1024
        )
        all_tests = parse_tests(output)  # type: ignore
        valid_tests = [test for test in all_tests if py_is_syntax_valid(test)]

        # n = 3
        # first_n = min(len(valid_tests), n)
        # valid_tests = valid_tests[:first_n]
        return GenerateResult(
            PY_TEST_GENERATION_CHAT_INSTRUCTION,
            message,
            output,
            sample_n_random(valid_tests, max_num_tests)
        )


DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res


def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))


def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)


def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """

    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f"{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}"
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = (
                handle_first_line_indent
                if cur_state == 0
                else handle_entire_body_indent
            )
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body

    return parse_indent_rec(func_body, 0)


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(1))
def gpt_chat(
    model: str,
    system_message: str,
    user_message: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)
