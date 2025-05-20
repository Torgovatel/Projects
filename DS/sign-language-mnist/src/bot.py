import asyncio
import sys
import aiohttp
import os
from aiogram import Bot, Dispatcher, types
from aiogram.types import (
    ContentType,
    ReplyKeyboardMarkup,
    KeyboardButton,
    BufferedInputFile,
)
from aiogram.filters import CommandStart
from common.logger import Logger
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(sys.argv[0], *[os.pardir] * 2))
LOG_DIR_PATH = os.path.abspath(os.path.join(BASE_DIR, os.getenv("LOG_DIR_PATH")))
LOG_FLNAME_TG_BOT = os.getenv("LOG_FLNAME_TG_BOT")

HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
API_PATH_PREDICTION = os.getenv("API_PATH_PREDICTION")
TOKEN = os.getenv("TG_BOT_TOKEN")
FASTAPI_URL = f"http://{HOST}:{PORT}/{API_PATH_PREDICTION}"

logger = Logger(LOG_FLNAME_TG_BOT, LOG_DIR_PATH)

bot = Bot(token=TOKEN)
dp = Dispatcher()

user_states = {}

keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="/predict"), KeyboardButton(text="/probabilities")],
        [KeyboardButton(text="/help"), KeyboardButton(text="/info")],
        [KeyboardButton(text="/clear")],
    ],
    resize_keyboard=True,
)

COMMAND_LIST_STR = """Список команд:
    /help - справка
    /info - информация о жестах
    /predict - классификация изображения с жестом
    /probabilities - получение полного списка вероятностей соответствия жеста букве
    /clear - сброс командного состояния"""


@dp.message(CommandStart())
async def start_handler(message: types.Message):
    user_id = message.chat.id
    logger.info({
        "user_id": user_id,
        "command": "start"
    })
    await message.answer(
        f"Добро пожаловать!\n\n{COMMAND_LIST_STR}",
        reply_markup=keyboard,
    )
    user_states[user_id] = None


@dp.message(lambda msg: msg.text and msg.text.lower() == "/info")
async def info_handler(message: types.Message):
    user_id = message.chat.id
    image_path = os.path.join(BASE_DIR, "src", "static", "american_sign_language.PNG")
    if os.path.exists(image_path):
        with open(image_path, "rb") as file:
            image_bytes = file.read()
            image_file = BufferedInputFile(
                image_bytes, filename="american_sign_language.PNG"
            )
            logger.info({
                "user_id": user_id,
                "command": "info",
                "error": False
            })
            await message.answer_photo(
                photo=image_file,
                caption="Для классификации используется американский язык жестов:",
                reply_markup=keyboard,
            )
    else:
        logger.warning({
                "user_id": user_id,
                "command": "info",
                "error": True
            })
        await message.answer(
            "Изображение с языком жестов не найдено 😓", reply_markup=keyboard
        )


@dp.message(
    lambda msg: msg.text
    and (msg.text.lower() == "/predict" or msg.text.lower() == "/probabilities")
)
async def choose_command(message: types.Message):
    user_id = message.chat.id
    user_states[user_id] = message.text.lower()
    logger.info({
        "user_id": user_id,
        "command": user_states[user_id]
    })
    await message.answer(
        "Теперь отправьте изображение для классификации.", reply_markup=keyboard
    )


@dp.message(lambda msg: msg.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message):
    user_id = message.chat.id
    log_dict = {
        "user_id": user_id,
        "command": user_states[user_id]
    }
    if user_states.get(user_id) is None:
        log_dict["error"] = "Not found previous command"
        logger.error(log_dict)
        await message.answer(
            "Прежде - выберите действие: /predict или /probabilities.",
            reply_markup=keyboard,
        )
        return

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}"
    logger.info(f"User {user_id} sent a photo for classification.")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(file_url) as resp:
                if resp.status != 200:
                    log_dict["error"] = "Failed to download image"
                    logger.error(log_dict)
                    await message.answer(
                        "Не удалось загрузить изображение 😓", reply_markup=keyboard
                    )
                else:
                    image_bytes = await resp.read()
                    data = aiohttp.FormData()
                    data.add_field(
                        "image",
                        image_bytes,
                        filename="image.jpg",
                        content_type="image/jpeg",
                    )
                    async with session.post(FASTAPI_URL, data=data) as fastapi_resp:
                        if fastapi_resp.status != 200:
                            log_dict["error"] = "Error during processing image"
                            logger.error(log_dict)
                            await message.answer(
                                "Во время обработки произошла ошибка 😓",
                                reply_markup=keyboard,
                            )
                        else:
                            result = await fastapi_resp.json()
                            prediction = result.get("prediction", "Неизвестно")
                            probabilities = result.get("probabilities", {})
                            response_text = f"Предсказание: {prediction}"
                            log_dict["error"] = False
                            log_dict["result"] = probabilities
                            logger.info(log_dict)
                            del log_dict["result"]

                            if user_states[user_id] == "/probabilities":
                                probabilities = result.get("probabilities", {})
                                response_text += "\n\nВероятности:\n"
                                response_text += "\n".join(
                                    [
                                        f"{key}: {float(value):.2%}"
                                        for key, value in probabilities.items()
                                    ]
                                )
                            await message.answer(response_text, reply_markup=keyboard)
        except aiohttp.ClientError as e:
            log_dict["error"] = "Aiohttp client error"
            logger.error(log_dict)
            await message.answer(
                "Сервер временно не доступен, повторите попытку позже 😓",
                reply_markup=keyboard,
            )


@dp.message(lambda msg: msg.text and msg.text.lower() == "/clear")
async def clear_handler(message: types.Message):
    user_id = message.chat.id
    logger.info({
        "user_id": user_id,
        "command": "clear"
    })
    user_states[user_id] = None
    await message.answer(
        "Команда сброшена 🫠", reply_markup=keyboard
    )


@dp.message(lambda msg: msg.text and msg.text.lower() == "/help")
async def help_handler(message: types.Message):
    user_id = message.chat.id
    logger.info({
        "user_id": user_id,
        "command": "help"
    })
    await message.answer(
        COMMAND_LIST_STR,
        reply_markup=keyboard,
    )


async def main():
    logger.info("STARTUP")
    logger.info("Loading completed")
    await dp.start_polling(bot)
    logger.info("SHUTDOWN")


if __name__ == "__main__":
    asyncio.run(main())
