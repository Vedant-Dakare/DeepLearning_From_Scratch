import json
import traceback
from PyPDF2 import PdfReader


def read_file(file):
    """
    Reads PDF or TXT files and returns extracted text.
    """

    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PdfReader(file)
            text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

            return text

        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    elif file.name.endswith(".txt"):
        try:
            return file.read().decode("utf-8")

        except Exception as e:
            raise Exception(f"Error reading TXT file: {str(e)}")

    else:
        raise Exception(
            "Unsupported file format. Only PDF and TXT files are supported."
        )


def get_table_data(quiz_str):
    """
    Converts the JSON quiz string returned by the LLM
    into a list of dictionaries that can be displayed
    as a Pandas DataFrame.
    """

    try:
        # If the model returns markdown JSON blocks, clean them
        quiz_str = quiz_str.strip()

        if quiz_str.startswith("```json"):
            quiz_str = quiz_str.replace("```json", "").replace("```", "").strip()

        elif quiz_str.startswith("```"):
            quiz_str = quiz_str.replace("```", "").strip()

        # Convert JSON string to Python dictionary
        quiz_dict = json.loads(quiz_str)

        quiz_table_data = []

        for key, value in quiz_dict.items():

            mcq = value.get("mcq", "")

            options = " || ".join(
                [
                    f"{option} -> {option_value}"
                    for option, option_value in value.get("options", {}).items()
                ]
            )

            correct = value.get("correct", "")

            quiz_table_data.append(
                {
                    "MCQ": mcq,
                    "Choices": options,
                    "Correct": correct,
                }
            )

        return quiz_table_data

    except Exception as e:
        print("\n========== JSON PARSING ERROR ==========")
        print("Error:", e)
        print("\nRaw Quiz Output:\n")
        print(quiz_str)
        print("\n========================================\n")

        traceback.print_exception(type(e), e, e.__traceback__)

        return None