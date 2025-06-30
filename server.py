import asyncio
import pickle
from user_input.input_model import Input_Model
import websockets
from cryptography.fernet import Fernet
import os
import traceback

from aggregation.aggregation import aggregate_weights
from security.cryptography import encrypt_obj, decrypt_obj
from utils.utils import save_model_weights, save_trained_model

HOST = '0.0.0.0'
PORT = 8000

NUM_CLIENTS = 5
client_counter = 0

FILE_PATH = "./user_input/data_processing_and_training.py"

global_model = Input_Model()
print("Global Model")
print(global_model.summary())

# Ensure the 'received_weights' directory exists
if not os.path.exists("./received_weights"):
    os.makedirs("./received_weights")

# Key generation
fixed_key = b'KLIHOhHpsJn3wVmaES9kTu6TJH8UmkhVhYcFd6QliDo='
cipher_suite = Fernet(fixed_key)

async def send_model(websocket, file):
    try:
        model = Input_Model()
        serialized_model = pickle.dumps(model)
        encrypted_model = encrypt_obj(cipher_suite, serialized_model)

        await websocket.send(str(len(encrypted_model)).encode())
        chunk_size = 1024
        for i in range(0, len(encrypted_model), chunk_size):
            await websocket.send(encrypted_model[i:i + chunk_size])

        print("Model sent successfully.")

        with open(file, 'rb') as f:
            file_data = f.read()
        encrypted_file = encrypt_obj(cipher_suite, file_data)

        await websocket.send(str(len(encrypted_file)).encode())
        for i in range(0, len(encrypted_file), chunk_size):
            await websocket.send(encrypted_file[i:i + chunk_size])

        print("File sent successfully.")

    except Exception as e:
        print("Error sending model or file:")
        traceback.print_exc()

async def receive_data_from_client(websocket):
    global client_counter
    try:
        client_id = await websocket.recv()
        print(f"Received client ID: {client_id}")

        encrypted_weights_size = int(await websocket.recv())
        print(f"Expected encrypted weights size: {encrypted_weights_size} bytes")

        encrypted_weights = bytearray()
        while len(encrypted_weights) < encrypted_weights_size:
            chunk = await websocket.recv()
            if isinstance(chunk, bytes):
                encrypted_weights.extend(chunk)

        decrypted_weights = decrypt_obj(cipher_suite, bytes(encrypted_weights))
        client_model_weights = pickle.loads(decrypted_weights)
        print(f"Received and decrypted model weights from client {client_id}")

        save_model_weights(client_id, client_model_weights)
        client_counter += 1

        # ✅ Send acknowledgment here
        await websocket.send("Weights received successfully.")

    except websockets.exceptions.ConnectionClosedOK:
        print("Client closed connection cleanly.")
    except Exception as e:
        print("Error in handling client:")
        traceback.print_exc()

async def handle_client(websocket):
    try:
        await send_model(websocket, FILE_PATH)
        await receive_data_from_client(websocket)

        if client_counter == NUM_CLIENTS:
            aw = aggregate_weights("./received_weights")
            save_trained_model(global_model, aw)
            print("All clients' data received and aggregated. Server will stop now.")
            await stop_server()

    except Exception as e:
        print("Error in handling client:")
        traceback.print_exc()

async def stop_server():
    print("Stopping server...")
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("Server stopped gracefully.")

async def start_server():
    try:
        async with websockets.serve(handle_client, HOST, PORT):
            print(f"Server listening on ws://{HOST}:{PORT}")
            await asyncio.Future()
    except asyncio.CancelledError:
        print("Server task cancelled. Shutting down.")
    finally:
        print("Server shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down.")
