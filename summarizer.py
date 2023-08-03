# LangChain Text Summarizer
# Based on https://python.langchain.com/docs/use_cases/summarization
# Example usage:
# python3 summarizer.py "https://www.forbes.com/sites/bryanrobinson/2020/09/02/why-neuroscientists-say-boredom-is-good-for-your-brains-health/"

import argparse
import dotenv

dotenv.load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some URLs.')
parser.add_argument('url', type=str, help='The URL to process')
args = parser.parse_args()

loader = WebBaseLoader(args.url)
docs = loader.load()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
chain = load_summarize_chain(llm, chain_type="stuff")

# Run the summarization and store the result in a variable
result = chain.run(docs)

# Print the result
print(result)
