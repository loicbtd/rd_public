import onnxruntime_genai as og

model_path = '/Users/loicbtd/Repo/microsoft_Phi_4_mini_instruct_onnx-gpu_gpu_int4_rtn_block_32'
config = og.Config(model_path)
model = og.Model(config)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

functions = '[{"name": "booking_flight_tickets", "description": "booking flights", "parameters": {"origin_airport_code": {"description": "The name of Departure airport code", "type": "string"}, "destination_airport_code": {"description": "The name of Destination airport code", "type": "string"}, "departure_date": {"description": "The date of outbound flight", "type": "string"}, "return_date": {"description": "The date of return flight", "type": "string"}}}, {"name": "booking_hotels", "description": "booking hotel", "parameters": {"destination": {"description": "The name of the city", "type": "string"}, "check_in_date": {"description": "The date of check in", "type": "string"}, "checkout_date": {"description": "The date of check out", "type": "string"}}}]'

lark_grammar = f"""start: TEXT | fun_call
TEXT: /[^{{](.|\n)*/
fun_call: <|tool_call|> %json {{"anyOf": {functions} }} <|/tool_call|>"""

params = og.GeneratorParams(model)
search_options = {}
search_options['max_length'] = 4096
search_options['temperature'] = 0.00001
search_options['top_p'] = 1.0
search_options['do_sample'] = False
params.set_search_options(**search_options)
params.set_guidance("lark_grammar", lark_grammar)
generator = og.Generator(model, params)

system_message_content = "You are a helpful assistant with these tools."
user_message_content = "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , then book hotel from 2025-12-04 to 2025-12-10 in Paris"

messages = f"""[{{"role": "system", "content": "{system_message_content}", "tools": {functions}}},{{"role": "user", "content": "{user_message_content}"}}]"""
prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
tokens = tokenizer.encode(prompt)
generator.append_tokens(tokens)

while not generator.is_done():
    generator.generate_next_token()
    
    new_token = generator.get_next_tokens()[0]
    
    decoded_token = tokenizer_stream.decode(new_token)
    
    print(decoded_token, end='', flush=True)
    
    if new_token == 200025:  # ID du token <|tool_call|>
        print(f"[DEBUG: Tool call Start token detected!]")
        
    if new_token == 200026:  # ID du token <|tool_call|>
        print(f"\n[DEBUG: Tool call End token detected!]")
