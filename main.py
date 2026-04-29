import asyncio
from visualize import App


async def main():
    app = App()
    while app._step():
        await asyncio.sleep(0)


asyncio.run(main())
