"""Prompt templates (paper appendix "Pretrained Language Models"; CONFIG F-7). Every template asks for a single Yes/No answer.

All three templates now use the same layout: instruction, blank line, graph, blank line, question block.
The serialization text starts with a space (serialize.py token rule), so the graph block begins " ...".
Question-first variants (E8) put the question block before the graph; the answer instruction stays last.
"""

INSTR = {
    "T1": "Here is an undirected graph. Every node is colored {colors}.",
    "T2": "You are given an undirected graph whose nodes are colored {colors}.",
    "T3": "Graph description (undirected; each node is {colors}):",
}
QBLOCK = {
    "T1": "Question: {question}",
    "T2": "{question}",
    "T3": "Using only the graph above, answer the following question.\n{question}",
}
QBLOCK_FIRST = {          # E8 question-first: "the graph below"
    "T1": "Question: {question}",
    "T2": "{question}",
    "T3": "Using only the graph below, answer the following question.\n{question}",
}
ANSWER = {
    "T1": "Answer with Yes or No.",
    "T2": "Reply with a single word: Yes or No.",
    "T3": "Respond \"Yes\" or \"No\".",
}
TEMPLATES = list(INSTR)


def color_phrase(colors):
    colors = list(colors)
    return " or ".join(colors) if len(colors) == 2 else ", ".join(colors[:-1]) + " or " + colors[-1]


def render(template_id, graph_text, question, colors=("red", "blue"), question_first=False):
    instr = INSTR[template_id].format(colors=color_phrase(colors))
    if question_first:
        q = QBLOCK_FIRST[template_id].format(question=question)
        return f"{instr}\n\n{q}\n\n{graph_text}\n\n{ANSWER[template_id]}"
    q = QBLOCK[template_id].format(question=question)
    return f"{instr}\n\n{graph_text}\n\n{q}\n{ANSWER[template_id]}"
