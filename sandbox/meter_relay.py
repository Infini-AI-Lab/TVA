#!/usr/bin/env python3
"""The prover's one way out: a TCP relay from the task's sealed network to the credit meter.

  meter_relay.py            (in the `meter` service of a task's environment/docker-compose.yaml)

A task's prover container (`main`) sits on `sealed`, a Docker network with no route anywhere: not the internet,
not the host. The `meter` service is the only other member, and it is also on an ordinary network. It accepts
connections on LISTEN_PORT and forwards each one, byte for byte, to TARGET_HOST:TARGET_PORT, the meter
(meter.py) the run script started on the host. Nothing else is reachable through it: the target is fixed when the
container starts. Without a target (the oracle agent, which needs no model) it accepts nothing. Standard library.
"""
import asyncio
import os
import sys

LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8080"))
TARGET_HOST = os.environ.get("TARGET_HOST", "")
TARGET_PORT = int(os.environ.get("TARGET_PORT", "0") or 0)


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(client_r, client_w):
    try:
        meter_r, meter_w = await asyncio.open_connection(TARGET_HOST, TARGET_PORT)
    except OSError as e:
        print(f"meter_relay: cannot reach the meter {TARGET_HOST}:{TARGET_PORT}: {e}", file=sys.stderr, flush=True)
        client_w.close()
        return
    await asyncio.gather(pipe(client_r, meter_w), pipe(meter_r, client_w))


async def main():
    if not TARGET_HOST or not TARGET_PORT:
        print("meter_relay: no meter given (TVA_METER_HOST, TVA_METER_PORT): relaying nothing", flush=True)
        await asyncio.Event().wait()
    srv = await asyncio.start_server(handle, "0.0.0.0", LISTEN_PORT)
    print(f"meter_relay: :{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}", flush=True)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
