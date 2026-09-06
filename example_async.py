"""Submit a second request while the first request is still decoding."""

import argparse
import asyncio
from pathlib import Path

from nanovllm import AsyncLLMEngine, SamplingParams


async def consume(
    name,
    stream,
    inject_event: asyncio.Event | None = None,
    inject_after: int | None = None,
):
    outputs = []
    async for output in stream:
        outputs.append(output)
        print(
            f"{name}: token={output.token_id} count={len(output.token_ids)} "
            f"finished={output.finished}"
        )
        if inject_after is not None and len(outputs) == inject_after:
            inject_event.set()
    return outputs[-1].text


async def main(model: Path) -> None:
    inject_event = asyncio.Event()
    engine = AsyncLLMEngine(
        str(model),
        enforce_eager=True,
        gpu_memory_utilization=0.65,
        max_model_len=512,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        scheduling_policy="decode_first",
    )

    first = await engine.submit(
        "Explain continuous batching in one sentence.",
        SamplingParams(temperature=0.1, max_tokens=16, ignore_eos=True),
    )
    first_task = asyncio.create_task(
        consume("first", first, inject_event=inject_event, inject_after=4)
    )

    await inject_event.wait()
    second = await engine.submit(
        "What is a KV cache?",
        SamplingParams(temperature=0.1, max_tokens=8, ignore_eos=True),
    )
    second_task = asyncio.create_task(consume("second", second))

    first_text, second_text = await asyncio.gather(first_task, second_task)
    await engine.close()
    print(f"first result: {first_text!r}")
    print(f"second result: {second_text!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    asyncio.run(main(parser.parse_args().model))
