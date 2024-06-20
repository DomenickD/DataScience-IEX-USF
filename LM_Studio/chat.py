# import streamlit as st
# from openai import OpenAI

# st.title("Code Assistant")
# st.write("Welcome to the Code Assistant app!")

# # Add a text input for user to enter code
# code = st.text_area("Enter your code here:")

# # Point to the local LM Studio server
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# # Add a button to submit the code
# if st.button("Submit"):
#     st.write("Processing your code...")

#     try:
#         # Send the code to the LM Studio server using the OpenAI client
#         completion = client.chat.completions.create(
#             model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
#             messages=[
#                 {"role": "system", "content": "You are a code assistant."},
#                 {"role": "user", "content": code}
#             ],
#             temperature=0.7,
#         )

#         # Display the response from the model
#         st.code(completion.choices[0].message.content)
#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# # Uncomment below for debugging purposes
# # completion = client.chat.completions.create(
# #   model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
# #   messages=[
# #     {"role": "system", "content": "Always answer in rhymes."},
# #     {"role": "user", "content": "Introduce yourself."}
# #   ],
# #   temperature=0.7,
# # )

# # print(completion.choices[0].message.content)

import streamlit as st
from openai import OpenAI
import time

# Initialize the OpenAI client pointing to the local LM Studio server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Title and introduction
st.title("Simple Chat with Code Assistant")
st.write("Welcome to the Code Assistant app!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process user input with LM Studio and get response
    with st.chat_message("assistant"):
        st.markdown("Processing your request...")

        try:
            # Send the code to the LM Studio server using the OpenAI client
            completion = client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "You are a code assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )

            # Get the response from the model
            response = completion.choices[0].message.content

            # Display assistant response in chat message container
            st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {e}")
