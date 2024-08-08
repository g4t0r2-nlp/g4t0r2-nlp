# Use the official Python 3.10 image
FROM python:3.10
 
# Set the working directory to /code
WORKDIR /code
 
# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Flask uygulamasını ve ilgili dizinleri kopyala
COPY static/ /code/static
COPY templates/ /code/templates

# Install requirements.txt 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
 
# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
 
# Set the working directory to the user's home directory
WORKDIR $HOME/code
 
# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/code

CMD ["gunicorn", "-b", "0.0.0.0:7860", "main:app"]