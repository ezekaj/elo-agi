"""Notebook Edit tool - Edit Jupyter notebooks programmatically."""

import json
import os


def notebook_edit(
    notebook_path: str,
    cell_number: int,
    new_source: str = "",
    edit_mode: str = "replace",
    cell_type: str = "code",
) -> str:
    """Edit a Jupyter notebook cell.

    edit_mode: 'replace', 'insert', 'delete'
    cell_type: 'code', 'markdown'
    """
    notebook_path = os.path.expanduser(notebook_path)

    if not os.path.exists(notebook_path):
        return f"Error: Notebook not found: {notebook_path}"

    try:
        with open(notebook_path, "r") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])

        if edit_mode == "delete":
            if cell_number < 0 or cell_number >= len(cells):
                return f"Error: Cell {cell_number} out of range (0-{len(cells)-1})"
            cells.pop(cell_number)
            nb["cells"] = cells
            with open(notebook_path, "w") as f:
                json.dump(nb, f, indent=1)
            return f"Deleted cell {cell_number} from {notebook_path}"

        # Build source lines
        source_lines = new_source.split("\n")
        source_list = [line + "\n" for line in source_lines[:-1]]
        if source_lines:
            source_list.append(source_lines[-1])

        new_cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_list,
        }
        if cell_type == "code":
            new_cell["outputs"] = []
            new_cell["execution_count"] = None

        if edit_mode == "replace":
            if cell_number < 0 or cell_number >= len(cells):
                return f"Error: Cell {cell_number} out of range (0-{len(cells)-1})"
            cells[cell_number]["source"] = source_list
            if cell_type != cells[cell_number].get("cell_type"):
                cells[cell_number]["cell_type"] = cell_type
        elif edit_mode == "insert":
            cells.insert(cell_number, new_cell)
        else:
            return f"Error: Unknown edit_mode: {edit_mode}"

        nb["cells"] = cells

        with open(notebook_path, "w") as f:
            json.dump(nb, f, indent=1)

        return f"Successfully {edit_mode}d cell {cell_number} in {notebook_path}"

    except json.JSONDecodeError:
        return f"Error: Invalid notebook format: {notebook_path}"
    except Exception as e:
        return f"Error editing notebook: {e}"
