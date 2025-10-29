import json
import random
import re

CUSTOM_PROMPTS = {69: 
f"""{{preamble}}
{{context}}

[Step-by-step instructions: You should follow these exactly.]

1. Identify the Basic Numerals (1 to 9). 
    - First, aim to fill out the forms for the numbers 1 through 9. You already have some of these provided but you need to work out the numbers for 3 (a), 7 (b) and 8 (c).
    - To work out 3, you should look at the Language X translations for 13 and 23. Notice that the they have a common substring. You can use the relationship between the number for 4 and 14, and 6 and 36 to help you identify the relationship.
    - To work out 7, you should apply the similar logic to work back from the number 67. The number 23 is to the number 3 as the number 67 is to the number 7.
    - To work out 8, you can do the same thing for 58.
    - Check that you have the correct values for 3 (a), 7 (b) and 8 (c).
2. Check for Patterns in the Teens (11 to 19).
    - Second, we will focus on the numbers in the teens. You have 11, 12, 13, 14 provided for you and you need to work out how 15 (d), 18 (e), and 19 (f) are formed.
    - Examine whether they share a common structure. In this case, realise that there is a fixed pattern from 1 to 11, 3 to 13, and 4 to 14. The teens have a common suffix.
    - Now, given you have the numbers for 5, 8, and 9, you can complete 15, 18, and 19.
3. Examine the compounding mechanism for larger numbers (Numbers like 23, 28, 36, 58, 67, etc.).
    - Given we know what the basic numbers 1 to 10 are, we can use this to work out how compounding numbers work. You want to aim for a formula to produce these numbers.
    - To work out the pattern, consider the following relationships:
        - The number for 2, the number for 10, the number for 3, and the number for 23
        - The numbers for 3, 10, 6 and 36.
        - The numbers for 5, 10, 8 and 58.
        - The numbers for 6, 10, 7 and 67.
    - Can you work out the common pattern for how these compound numbers work? Explicitly think about this rule and write it down.
    - Verify that this pattern holds by considering:
        - The numbers for 8, 10, 1 and 81.
        - The numbers for 9, 10, 2 and 92.
4. Fill in the compounding numbers given this rule.
    - Now that you know the rule for these numbers we can complete the other numbers that follow a similar pattern: 28 (g), 44 (h), 52 (i), 74 (k), and 99 (m).
    - For each number, explicitly write out each number's compound structure in terms of the rule.
    - Double-check that your answers follow the rule you established in step 3.
5. Finally, we need to work out how to deal with the numbers 60 and 80.
    - Realise that all you need to do is apply the rule you determined in step 3. This is simple because 0 is just the absence of a number in the 1s place, i.e. 60 is the same as 67 just without the + 7.
    - Use this to complete 60 (j) and 80 (l).
6. Double-check your answers before returning the structured json output.

[Questions: Answer each of the following questions.]
{{subquestions}}

Your final answer should be in json with the following format:
{{formatted_output}}""",

164:
                  
f"""{{preamble}}
{{context}}

[Step-by-step instructions: You should follow these exactly.]

1. Start with the person.
    - Most of the examples refer to the pronoun "we", so let's start with those. Looking at the Language X pronouns, we seem to be able to group them into two categories, based on the general structure. Aim to group them into these two groups.
    - Now let's aim to find out what the difference is between these groups. In order to do that, look at the situations provided. Can you find any differences in the meaning of the pronoun "we" that could explain why we need two different sets? This should be the first relevant distinction. We have two types of "we" and one type of "they".
2. Figuring out why there are multiple versions for the same person.
    - Now we notice that for each of the two "we" sets of pronouns, we have three different forms.
    - Try to group together the situations that use the same pronoun. For example, situations a and l use the same pronoun. What do these situations have in common, compared to, for example, situations b and k? Does this distinction also explain why situation i uses the same pronoun as situations a and l? Does it also explain why situations h and j use the same pronoun?
    - Once you figured out this distinction too, you're almost there.
3. Putting it all together.
    - Now that you know the two relevant distinctions from this language, try to make a table, in which each column represents one person (as you discovered in step 1), and each row represents the distinction found in step 2.
    - Try to fill in the table using situations a, b, c, d, e, f, g, and h.
    - Make sure your rule still holds true for situations i, j, k, l.
4. Going to the tasks
    - Now that you know the general rules, let's look at the tasks and aim to answer the questions. For situations 1-4, you are given three new words, which refer to "you". We have not had any occurrences of "you" before, so we do not know any of these words.
    - Nevertheless, if we look at the table you created at step 3, we can fit these into the table based on similarities with the other pronouns. In order to do so:
        - Create a new column that corresponds to the newly given "you" pronoun.
        - Attempt to place the three new pronouns given into that column, by looking at similarities with other columns. Is any of these three new pronouns very similar to one that you already have? If so, it is probably safe to place it on the same row. What about the other two new pronouns? Can you find one column that has pronouns very similar to the ones that you are given here?
    - For task 5, you are also given three pronouns. You already know two of them, but you have one that has not appeared before. Can you fit this one in the table as well?
    - Now that you know which pronoun corresponds to which situation, look at the tasks and choose the correct pronoun based on the table you created. In order to do so:
        - First, check the English pronoun given in the prompt in order to know which column of your table to look at.
        - Then, read the situation carefully and decided to which row of your table it best corresponds to.
        - Once you determined the row and column, use that pronoun to give your answer to each of the tasks 1-11.
5. Double-check your answers before returning the structured json output.

[Questions: Answer each of the following questions.]

{{subquestions}}

Your final answer should be in json with the following format:
{{formatted_output}}""",
}

def clean_whitespace(text: str) -> str:
    """Clean up excessive whitespace and normalize spacing."""
    if not text:
        return text
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple consecutive newlines with double newlines max
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing spaces from each line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    # Remove excessive spaces (more than 2 consecutive spaces, except in data tables)
    # But preserve tab characters and single/double spaces
    text = re.sub(r' {3,}', '  ', text)
    
    return text

def normalize_tables(text: str) -> str:
    """Normalize tab-separated data tables for consistent formatting."""
    if not text:
        return text
    
    lines = text.split('\n')
    normalized_lines = []
    
    for line in lines:
        # If line contains tabs (likely a data table), normalize spacing
        if '\t' in line:
            # Split by tab and rejoin with consistent spacing
            parts = line.split('\t')
            if len(parts) >= 2:
                # Ensure consistent spacing: term + 2 spaces + definition
                normalized_line = f"{parts[0].strip()}  {parts[1].strip()}"
                # If there are more parts, add them with single spaces
                for part in parts[2:]:
                    normalized_line += f" {part.strip()}"
                normalized_lines.append(normalized_line)
            else:
                normalized_lines.append(line)
        else:
            normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)

def strip_question_title(preamble: str) -> str:
    """Remove question titles/numbers/marks from the beginning of preamble text."""
    if not preamble:
        return preamble
    
    # Split into lines and process the first line
    lines = preamble.split('\n')
    first_line = lines[0].strip()
    
    # Patterns to match and remove:
    patterns = [
        r'^Question\s+\d+:?\s*.*?\s*\(\d+\s*marks?\)\s*$',  # "Question X: Language Y (Z marks)"
        r'^Question\s+\d+\s+.*?\s*\(\d+\s*marks?\)\s*$',    # "Question X Language Y (Z marks)" 
        r'^\d+\.\s+.*?\s*\(\d+\)\s*$',                      # "5. Title (20)"
        r'^.*?\s*\(\d+\s*marks?\)\s*$',                     # "Language X (25 marks)"
    ]
    
    for pattern in patterns:
        if re.match(pattern, first_line, re.IGNORECASE):
            # Remove the first line if it matches a title pattern
            remaining_lines = lines[1:]
            # Also remove empty lines at the beginning
            while remaining_lines and not remaining_lines[0].strip():
                remaining_lines = remaining_lines[1:]
            return '\n'.join(remaining_lines)
    
    return preamble

def load_questionsheet(qsheet: dict, no_context: bool = False):
    subquestions = json.loads(qsheet["questions"])

    all_subquestions = ""
    for sq in subquestions:
        all_subquestions += f"\n{sq['prompt']}\n"
        for sp in sq["subprompts"]:
            all_subquestions += f"{sp['questionpart_n']} {sp['question']}"
            all_subquestions += "\n"

    # Strip question title from preamble and clean whitespace
    clean_preamble = strip_question_title(qsheet['preamble'])
    clean_preamble = clean_whitespace(clean_preamble)
    
    # Clean context if present
    clean_context = clean_whitespace(qsheet['context']) if qsheet.get('context') else ""
    clean_context = normalize_tables(clean_context) if clean_context else ""
    
    if no_context:
        prompt = f"{clean_preamble}\n\n{all_subquestions}"
    else:
        prompt = f"{clean_preamble}\n{clean_context}\n\n{all_subquestions}"

    return prompt


def format_answers(questionpart_ns: list[str], answers: list[str], randomize=False):
    formatted_output = {}
    formatted_answers = {}
    formatted_answers_train = {}
    if randomize:
        raise NotImplementedError("Randomization is not implemented")
        data = [(q, a) for q, a in zip(questionpart_ns, answers)]
        random.shuffle(data)
        questionpart_ns = [d[0] for d in data]
        answers = [d[1] for d in data]
    for i, qn in enumerate(questionpart_ns):
        formatted_output[qn] = ""

        try:
            ans = json.loads(answers[i])
        except:
            ans = answers[i]
        if isinstance(ans, list):
            formatted_answers_train[qn] = random.choice(ans)

        else:
            formatted_answers_train[qn] = ans

        formatted_answers[qn] = answers[i]

    formatted_output = json.dumps(formatted_output)

    return formatted_output, formatted_answers, formatted_answers_train


def load_question_only(qsheet: dict, question_index: int):
    subquestions = json.loads(qsheet["questions"])
    sq = subquestions[question_index]
    sq["metadata"] = qsheet
    # sq['preamble'] = qsheet["preamble"]
    # sq['context'] = qsheet["context"]
    return sq


def format_question(
    question_entry: dict,
    model: str,
    no_context: bool = False,
    cot: bool = False,
    custom_prompt: bool = False,
):
    if "question_details" not in question_entry:
        raise KeyError(f"Missing 'question_details' key in question entry. Available keys: {list(question_entry.keys())}")
    sq = question_entry["question_details"]
    all_subquestions = ""
    questionpart_ns = []
    answers = []

    # Only include the specific subquestion being asked, not all subquestions
    all_subquestions += f"\n{sq['prompt']}\n"
    for sp in sq["subprompts"]:
        all_subquestions += f"{sp['questionpart_n']} {sp['question']}"
        questionpart_ns.append(sp["questionpart_n"])
        answers.append(sp["answer"])
        all_subquestions += "\n"

    formatted_output, formatted_answers, formatted_answers_train = format_answers(
        questionpart_ns, answers
    )

    if custom_prompt:
        prompt = CUSTOM_PROMPTS[sq["metadata"]["overall_question_n"]].format(preamble=sq["metadata"]['preamble'], context=sq["metadata"]['context'], subquestions=all_subquestions,formatted_output=formatted_output)

    else:
        # Get background context without all the other subquestions
        # Strip question title from preamble and clean whitespace
        clean_preamble = strip_question_title(sq["metadata"]['preamble'])
        clean_preamble = clean_whitespace(clean_preamble)
        
        # Clean context if present
        clean_context = clean_whitespace(sq["metadata"]['context']) if sq["metadata"].get('context') else ""
        clean_context = normalize_tables(clean_context) if clean_context else ""
        
        if no_context:
            background = clean_preamble
        else:
            background = f"{clean_preamble}\n{clean_context}" if clean_context else clean_preamble

        if cot:
            instructions = "Think step by step about your answer for each part of the question. Then write your final answer with json output with the keys as provided below:"
        else:
            instructions = "Make sure to finish your answer with json output with the keys as provided below:"

        # Clean the subquestions as well
        clean_subquestions = clean_whitespace(all_subquestions)

        # Format: background + specific question only (no preamble, no "Now respond to the following questions")
        prompt = f"{background}\n\n{clean_subquestions}\n\n{instructions}\n{formatted_output}"

    with open("../data/model_list.json") as f:
        model_list = json.load(f)
        header = model_list[model]["chat_header"]
        footer = model_list[model]["chat_footer"]

        # Apply final cleanup to the entire prompt
        prompt = clean_whitespace(header + prompt + footer)

    return (
        prompt,
        formatted_answers,
        formatted_answers_train,
        question_entry["index"],
        question_entry,
    )


def load_question(
    qsheet: dict,
    question_index: int,
    model: str,
    no_context: bool = False,
    cot: bool = False,
):
    subquestions = json.loads(qsheet["questions"])
    sq = subquestions[question_index]

    all_subquestions = ""
    questionpart_ns = []
    answers = []
    all_subquestions += f"\n{sq['prompt']}\n"
    for sp in sq["subprompts"]:
        all_subquestions += f"{sp['questionpart_n']} {sp['question']}"
        questionpart_ns.append(sp["questionpart_n"])
        answers.append(sp["answer"])
        all_subquestions += "\n"

    formatted_output, formatted_answers, _ = format_answers(questionpart_ns, answers)

    # Get background context without all the other subquestions
    # Strip question title from preamble and clean whitespace
    clean_preamble = strip_question_title(qsheet['preamble'])
    clean_preamble = clean_whitespace(clean_preamble)
    
    # Clean context if present
    clean_context = clean_whitespace(qsheet['context']) if qsheet.get('context') else ""
    clean_context = normalize_tables(clean_context) if clean_context else ""
    
    if no_context:
        background = clean_preamble
    else:
        background = f"{clean_preamble}\n{clean_context}" if clean_context else clean_preamble

    if cot:
        instructions = "Think step by step about your answer for each part of the question:"
    else:
        instructions = "Only respond with json output. Do not include anything other than the json in your response. Format your response as a json file with the keys as provided below:"

    # Clean the subquestions as well
    clean_subquestions = clean_whitespace(all_subquestions)

    # Format: background + specific question only (no preamble, no "Now respond to the following questions")
    prompt = f"{background}\n\n{clean_subquestions}\n\n{instructions}\n{formatted_output}"

    with open("../data/model_list.json") as f:
        model_list = json.load(f)
        header = model_list[model]["chat_header"]
        footer = model_list[model]["chat_footer"]

        # Apply final cleanup to the entire prompt
        prompt = clean_whitespace(header + prompt + footer)

    return prompt, formatted_answers


def load_all_questions_only(
    question_sheets: list[dict],
):
    indices = []
    as_is = []
    for qsheet in question_sheets:
        for i in range(len(json.loads(qsheet["questions"]))):
            rest = load_question_only(qsheet, i)
            as_is.append(rest)
            indices_entry = []
            indices_entry.append(qsheet["overall_question_n"])
            indices_entry.append(qsheet["obfuscated_question_n"])
            indices_entry.append(qsheet["obfuscated"])
            indices_entry.append(qsheet["obf_num"])
            indices_entry.append(rest["question_n"])
            indices.append(indices_entry)

    return indices, as_is


def load_flattened_questions(
    question_sheets: list[dict],
    model: str,
    no_context: bool = False,
    cot: bool = False,
    custom_prompt: bool = False
):
    questions, correct_answers, answers_train, q_idx, metadata = [], [], [], [], []

    for ques_entry in question_sheets:
        qu, ca, cat, qi, meta = format_question(
            ques_entry, model=model, no_context=no_context, cot=cot, custom_prompt=custom_prompt
        )
        questions.append(qu)
        correct_answers.append(ca)
        answers_train.append(cat)
        metadata.append(meta)
        q_idx.append(qi)

    return questions, correct_answers, answers_train, q_idx, metadata


def load_all_questions(
    question_sheets: list[dict],
    model: str,
    no_context: bool = False,
    cot: bool = False,
):
    prompts = []
    answers = []
    indices = []
    as_is = []
    for qsheet in question_sheets:
        for i in range(len(json.loads(qsheet["questions"]))):
            prompt, answer = load_question(qsheet, i, model, no_context, cot)
            prompts.append(prompt)
            answers.append(answer)
            rest = load_question_only(qsheet, i)
            as_is.append(rest)
            indices_entry = []
            indices_entry.append(qsheet["overall_question_n"])
            indices_entry.append(qsheet["obfuscated_question_n"])
            indices_entry.append(qsheet["obfuscated"])
            indices_entry.append(qsheet["obf_num"])
            indices_entry.append(qsheet["question_n"])
            indices.append(indices_entry)

    return prompts, answers, indices, as_is