import subprocess
import time

def run_curl(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    print("Output:", output.decode())
    if error:
        print("Error:", error.decode())
    print("-" * 50)

# Start game
start_command = '''curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "start", "user_id": null}' '''
print("Starting game...")
run_curl(start_command)

time.sleep(2)  # Wait for question generation

# Submit contestant 1's answer
contestant1_command = '''curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "For me, a perfect first date would be urban exploration with our cameras - discovering hidden street art, trying a hole-in-the-wall restaurant we stumble upon, and capturing moments together. It shows creativity, spontaneity, and lets us create shared memories while seeing how we each view the world.", "user_id": 1}' '''
print("Submitting contestant 1's answer...")
run_curl(contestant1_command)

time.sleep(2)  # Wait for processing

contestant2_command = """curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "Id take you to mcdonalds for a kids meal and then back to my house. I think thats good enough for a first date. However i am a very loyal guy and i will only ever take you to mcdonalds", "user_id": 2}'"""
print("Submitting contestant 2's answer...")
run_curl(contestant2_command)
