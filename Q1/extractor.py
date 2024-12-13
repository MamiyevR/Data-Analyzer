import pandas as pd
import validators
import asyncio
import attr
from aiohttp import ClientSession
from tenacity import retry, wait_exponential, stop_after_attempt
from bs4 import BeautifulSoup
import json
from transformers import pipeline
import torch


# Check if the string is a valid URL
def is_string_an_url(url_string: str) -> bool:
    if not validators.url(url_string):
        return False
    return True


# read cvs into a dataframe by chunks and validate the URLs
def extract_data(file_path, chunksize=100):
    for chunk in pd.read_csv(
        file_path, chunksize=chunksize, usecols=["URL"], on_bad_lines="warn"
    ):
        chunk = chunk[chunk["URL"].apply(is_string_an_url)]
        yield chunk


def extract_text(html):
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Extract the title (fallback to h1 if title tag is missing)
        title = soup.title.string if soup.title else None
        if not title:
            h1_tag = soup.find("h1")
            title = h1_tag.get_text(strip=True) if h1_tag else "Untitled"

        # Extract the main content
        main_content = ""
        article = soup.find("article")
        if article:
            main_content = " ".join(
                p.get_text(" ", strip=True) for p in article.find_all("p")
            )
        else:
            main_content = " ".join(
                p.get_text(" ", strip=True) for p in soup.find_all("p")
            )

        # Ensure spaces between paragraphs
        main_content = " ".join(main_content.split())

        if not main_content:
            main_content = "No content found."

        return {"title": title, "content": main_content}
    except Exception as e:
        print(f"Error extracting text: {e}")
        return {"title": None, "content": None}


@attr.s
class Fetch:
    rate = attr.ib(default=10, converter=int)  # requests per second

    # retry decorator with exponential backoff
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3)
    )
    async def fetch_page(self, url, semaphore, session: ClientSession):
        # limit the rate of requests
        async with semaphore:
            try:
                # make GET request using session
                async with session.get(url, timeout=10) as response:
                    # return HTML content
                    html = await response.text()

                    # extract text from HTML
                    text = extract_text(html)

                    return {
                        "url": url,
                        "title": text["title"],
                        "content": text["content"],
                        "status": response.status,
                    }
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return {"url": url, "title": None, "content": None, "error": str(e)}


async def process_chunk(url_chunk, fetch, semaphore, session):
    tasks = []
    for url in url_chunk["URL"].to_list():
        # Create a task for each URL
        tasks.append(fetch.fetch_page(url=url, semaphore=semaphore, session=session))

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Return the results for this chunk
    return results


# Retry logic for summarization
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def summarize_blocking(pipe, content):
    # Blocking summarization function for retry logic.
    summary = pipe(content, max_length=250, min_length=100, do_sample=False)[0][
        "summary_text"
    ]
    return summary


async def summarize_data(pipe, content):
    # Run summarization in a separate thread to avoid blocking the event loop
    return await asyncio.to_thread(summarize_blocking, pipe, content)


async def summarize_batch(pipe, batch):
    # Asynchronously summarize a batch of data
    tasks = []
    for data in batch:
        if "status" not in data or data["status"] != 200:
            # Skip items with invalid status
            continue
        try:
            # Create a task for summarization
            task = summarize_data(pipe, data["content"])
            tasks.append((data, task))
        except Exception as e:
            print(f"Error preparing content for summarization: {e}")

    # Process all tasks concurrently
    summaries = []
    for data, task in tasks:
        try:
            summary = await task
            data["summary"] = summary
            del data["status"]
            summaries.append(data)
        except Exception as e:
            print(f"Error summarizing content from {data['url']}: {e}")
            data["summary"] = None
            summaries.append(data)

    return summaries


async def summarizer(json_file, batch_size=10):
    # check if the GPU is available
    device = 0 if torch.cuda.is_available() else -1

    # we are using the Google T5 model from HuggingFace for summarization
    # https://huggingface.co/google-t5/t5-small
    pipe = pipeline("summarization", model="google-t5/t5-small", device=device)

    # Read the JSON file containing fetched data
    with open(json_file, "r", encoding="utf-8") as infile:
        try:
            data_list = json.load(infile)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    tasks = []
    batch = []

    for data in data_list:
        batch.append(data)
        if len(batch) == batch_size:
            tasks.append(summarize_batch(pipe, batch))
            batch = []

    # Process remaining batch
    if batch:
        tasks.append(summarize_batch(pipe, batch))

    results = await asyncio.gather(*tasks)

    # Flatten results (since each batch returns a list of summaries)
    flattened_results = [item for sublist in results for item in sublist]

    # Write results to a json file
    with open("data/output.json", "w", encoding="utf-8") as final:
        json.dump(flattened_results, final, indent=2, ensure_ascii=False)


async def main(rate, limit):
    url_iterator = extract_data("data/input.csv")
    semaphore = asyncio.Semaphore(limit)

    fetch = Fetch(rate=rate)

    all_results = []
    # Create a ClientSession
    async with ClientSession() as session:
        for url_chunk in url_iterator:
            # Process each chunk of URLs
            results = await process_chunk(url_chunk, fetch, semaphore, session)

            all_results.extend(results)

            # Wait for the remaining time to complete 10 seconds
            await asyncio.sleep(1 / rate)

    # Write results to a json file
    with open("data/output.json", "w", encoding="utf-8") as final:
        json.dump(all_results, final, indent=2, ensure_ascii=False)

    # Summarize the content
    await summarizer("data/output.json")


# run the main function
asyncio.run(main(rate=10, limit=10))
