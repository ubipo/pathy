import ast, textwrap
import nbformat


DOC_FOOTER = textwrap.dedent("""\
    <hr>
    This documentation page was generate from a python file.
    The file is located in the repo folder corresponding to this 
    documentation page.
    You can also click the `suggest edit` link (GitHub logo, top right)
    to open the file in GitHub.
""")

def py_to_nb_node(py_content: str, ext: str = "py") -> nbformat.NotebookNode:
    if py_content is None:
        return None
    
    docstring = ast.get_docstring(ast.parse(py_content))
    if (docstring is None):
        return nbformat.NotebookNode(
            nbformat=nbformat.v4.nbformat,
            nbformat_minor=nbformat.v4.nbformat_minor,
            metadata={},
            cells=[],
        )
        
    docstring_lines = docstring.split("\n")
    title = docstring_lines[0]
    content = "\n".join(docstring_lines[2:])
    title_cell = nbformat.v4.new_markdown_cell(f"# {title}")
    content_cell = nbformat.v4.new_markdown_cell(f"{content}")
    info_cell = nbformat.v4.new_markdown_cell(DOC_FOOTER)
    nb = nbformat.NotebookNode(
        nbformat=nbformat.v4.nbformat,
        nbformat_minor=nbformat.v4.nbformat_minor,
        metadata={
            "title": title,
            "language_info": {
                "file_extension": ext
            }
        },
        cells=[title_cell, content_cell, info_cell],
    )
    return nb
