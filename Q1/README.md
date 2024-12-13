# Q1 Extractor

### Main libraries Used

- `pandas`: For reading and processing CSV files.
- `validators`: For URL validation.
- `asyncio`: For asynchronous programming.
- `aiohttp`: For making asynchronous HTTP requests.
- `tenacity`: For implementing retry logic with exponential backoff.
- `BeautifulSoup`: For parsing HTML and extracting content.
- `transformers`: For text summarization using pre-trained models.

### Code Explanation

#### URL Validation

The function `is_string_an_url` checks if a given string is a valid URL using the `validators` library.

#### Data Extraction

The function `extract_data` reads the CSV file in chunks and validates the URLs.

#### HTML Content Extraction

The function `extract_text` parse HTML into BeautifulSoup and extracts the title and main content from the HTML of a webpage. Meaningful content identified using semantic tags (<article\>, <p\>)

#### Fetching Web Pages

The `Fetch` class uses `aiohttp` to fetch web pages asynchronously with rate limiting and retry logic.

#### Processing Chunks

The function `process_chunk` processes a chunk of URLs concurrently.

#### Summarization

The function `summarize_data` uses a pre-trained model from HuggingFace to summarize the content. We used abstractive summarization using google-t5 model from HuggingFace. For faster and scalable summarization, rule based extractive summarization techniques can be used (e.g., selecting the most important sentences).

#### Main Function

The `main` function orchestrates the entire process, from reading the CSV file to fetching web pages and summarizing the content.

## Conclusion

The implemented Extractor processes URLs from a CSV file, extracts relevant information, and summarizes the content. The solution is scalable, handles rate limiting, and includes error handling and parallel processing to ensure efficiency and reliability.
