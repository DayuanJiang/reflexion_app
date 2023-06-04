
import os
import re

import streamlit as st
from streamlit_ace import st_ace
import pandas as pd
import openai

from py_executor import PyExecutor
from py_generate import PyGenerator

openai.api_key = os.environ.get("OPENAI_API_KEY")
st.set_page_config(layout="wide")

def extract_function_name(function_definition: str) -> str:
    """Extracts the function name from a function definition.

    Args:
        function_definition: A string containing a function definition.

    Returns:
        The extracted function name as a string.
    """
    match = re.search(r"def\s+(\w+)", function_definition)
    if match:
        return match.group(1)
    else:
        return None


@st.cache_data
def load_data():
    path = "reflexion_humaneval_py_pass_at_1.jsonl"
    jsonObj = pd.read_json(path_or_buf=path, lines=True)
    return jsonObj


st.title("GPT python function creator")
usage = """
## このアプリについて
このアプリは、GPT-4を使用してPython関数を生成するものです。  
まず、与えられた関数定義を基に内部テストケースを作成し、生成された関数がテストケースに通過するか判定します。  
通過しない場合は、GPT-4に反省(reflexion)してもらい,再度関数を生成してもらいます。  
このプロセスを繰り返し、テストケースに合格する関数を得ます。
実装する際にReflexionの[論文](https://arxiv.org/abs/2303.11366)とGithubの[コード](https://github.com/noahshinn024/reflexion/tree/main)を参考にしました。

## 使い方
デモではHumanEvalデータセットを使用しています。  
プルダウンメニューより関数例を選択できます。  
「Code Definition」に関数定義が、「Code tests」にテストケースが表示されます。  
「Submit」ボタンで関数生成が開始され、生成過程が右側に表示されます。  
最終結果は左欄に表示されます。

自分で関数を定義して生成することも可能です。  
その場合、「Code Definition」と「Code tests」に関数定義とテストケースを入力してください。  
「Code tests」には`def check(candidate):`以降のものを定義すれば良いです。
"""


col1, col2 = st.columns(2)

with col1:
    st.write(usage)
    data = load_data()
    # create a selectbox to choose a example from data
    example = st.selectbox(
        "Select an example from HumanEval", data["entry_point"].tolist()
    )

    data_row = data.iloc[data.entry_point.tolist().index(example)]
    function_definition = data_row.prompt
    code_tests_example = data_row.test

    st.markdown("## Code Definition")
    function_definition = st_ace(
        value=function_definition,
        language="python",
        auto_update=True
        # theme="monokai",
    )

    st.markdown("## Code tests")
    final_test_cases = st_ace(
        value=code_tests_example, language="python", auto_update=True
    )

    # create a submit button
    submit_button = st.button(label="Submit")

gen = PyGenerator()
exe = PyExecutor()


if submit_button:
    with col1:
        st.info("Generating internal test cases... Please wait a moment.")
    is_solved = False
    reflections = []
    implementations = []
    test_feedback = []
    cur_func_impl = ""
    max_iters = 5

    function_name = extract_function_name(function_definition)

    tests_i_result = gen.internal_tests(func_sig=function_definition, max_num_tests=5)
    tests_i = tests_i_result.processed_result
    
    with col2:
        st.write("## Generated internal test cases:")
        with st.spinner("Generating test cases..."):
            st.code(tests_i_result.generated_string, language="python")
            # show test cases
            with st.expander("See Generate Details"):
                st.info("System message: ")
                st.code(tests_i_result.system_message, language="raw")
                st.info("User message: ")
                st.code(tests_i_result.user_message, language="raw")
                st.success("Output: ")
                st.code(tests_i_result.generated_string, language="python")

        # first attempt
        st.write("## First attempt")
        with st.spinner("Generating first attempt..."):
            first_attempt_result = gen.func_impl(
                func_sig=function_definition, strategy="simple"
            )
            cur_func_impl = first_attempt_result.processed_result
            implementations.append(cur_func_impl)
            first_attempt_is_passing, first_attempt_feedback, _ = exe.execute(
                cur_func_impl, tests_i
            )
            test_feedback.append(first_attempt_feedback)

            st.success("Generated Function Implementation: ")
            st.code(first_attempt_result.processed_result, language="python")
            with st.expander("See Generate Details"):
                st.info("System message: ")
                st.code(first_attempt_result.system_message, language="raw")
                st.info("User message: ")
                st.code(first_attempt_result.user_message, language="raw")
                st.success("Output: ")
                st.code(first_attempt_result.generated_string, language="python")

        if first_attempt_is_passing:
            print_func = st.success
        else:
            print_func = st.error
        print_func(f"Internel test result: {first_attempt_is_passing}")
        st.code(first_attempt_feedback, language="raw")

        # if solved, exit early
        if first_attempt_is_passing:
            final_test_is_passing = exe.evaluate(
                function_name, cur_func_impl, final_test_cases, timeout=10
            )
            print_func = st.success if final_test_is_passing else st.error
            print_func(f"Final test result: {final_test_is_passing}")
            if final_test_is_passing:
                is_solved = True
                with col1:
                    st.write("## Solution")
                    st.code(cur_func_impl, language="python")

        # use self-reflection to iteratively improve
        cur_iter = 1
        cur_feedback = first_attempt_feedback
        while cur_iter < max_iters and not is_solved:
            # get self-reflection
            with st.spinner("Generating self-reflection..."):
                reflection_result = gen.self_reflection(
                    func=cur_func_impl, feedback=cur_feedback
                )
                reflections += [reflection_result.processed_result]

                st.write(f"## Self-reflection {cur_iter}")
                st.write(f"**:blue[{reflection_result.processed_result}]**")
                with st.expander("See Generate Details"):
                    st.info("System message: ")
                    st.code(reflection_result.system_message, language="raw")
                    st.info("User message: ")
                    st.code(reflection_result.user_message, language="raw")
                    st.success("Output: ")
                    st.code(reflection_result.generated_string, language="python")

            # apply self-reflection in the next attempt
            with st.spinner("Generating next attempt..."):
                cur_func_impl_result = gen.func_impl(
                    func_sig=function_definition,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection_result.processed_result,
                )
                cur_func_impl = cur_func_impl_result.processed_result
                implementations.append(cur_func_impl)

                st.success("Next Attemptation")
                st.code(cur_func_impl_result.processed_result, language="python")
                with st.expander("See Generate Details"):
                    st.write("## System message: ")
                    st.write(cur_func_impl_result.system_message)
                    st.write("## User message: ")
                    st.write(cur_func_impl_result.user_message)
                    st.write("## Output: ")
                    st.write(f"**:blue[{cur_func_impl_result.generated_string}]**")

                # check if all internal unit tests pass
                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                if is_passing:
                    print_func = st.success
                else:
                    print_func = st.error
                print_func(f"Next attempt internal test result: {is_passing}")
                st.code(cur_feedback, language="raw")

            # if solved, check if it passes the real tests, exit early
            if is_passing or cur_iter == max_iters - 1:
                final_test_is_passing = exe.evaluate(
                    function_name, cur_func_impl, final_test_cases, timeout=10
                )

                print_func = st.success if final_test_is_passing else st.error
                print_func(f"Final test result: {final_test_is_passing}")
                if final_test_is_passing:
                    is_solved = True
                    with col1:
                        st.write("## Solution")
                        st.code(cur_func_impl, language="python")

            cur_iter += 1

    if not is_solved:
        with col1:
            st.error("Failed to generate a solution.")
