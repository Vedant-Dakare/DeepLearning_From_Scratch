from setuptools import setup, find_packages

setup(
    name="mcqgen",
    version="0.0.1",
    author="Vedant Dakare",
    author_email="dakarevedant24@gmail.com",
    install_requires=["openai", "langchain", "streamlit", "PyPDF2", "python-dotenv"],
    packages=find_packages(),
)