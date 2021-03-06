import nbformat


def py_to_nb_node(py_content: str, fmt: str = "py") -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    docstring_cell = nbformat.v4.new_markdown_cell("# Testing markdown")
    nb.cells.append(docstring_cell)
    return nb
