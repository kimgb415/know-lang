# Step 1: Choose a base Python image
FROM python:3.10-slim

# Step 2: Set a working directory
WORKDIR /home/user/app

# Step 3: Copy all your code into the container
COPY . /home/user/app

# Step 4: Install dependencies
#         - If you rely on pyproject.toml (poetry or setuptools):
RUN pip install --no-cache-dir .

# Step 5: Expose a port (optional, helps clarity, though HF Spaces auto-detects)
EXPOSE 7860

# Step 6: Command to run your Gradio app (or any Python script)
CMD ["python", "app.py"]