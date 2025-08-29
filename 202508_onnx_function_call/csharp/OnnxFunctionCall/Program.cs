using Microsoft.ML.OnnxRuntimeGenAI;

string modelPath = "/Users/loicbtd/Repo/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4";

using Config config = new(modelPath);
using Model model = new(config);
using Tokenizer tokenizer = new(model);
using TokenizerStream tokenizerStream = tokenizer.CreateStream();

string functions =
    "[{\"name\": \"booking_flight_tickets\", \"description\": \"booking flights\", \"parameters\": {\"origin_airport_code\": {\"description\": \"The name of Departure airport code\", \"type\": \"string\"}, \"destination_airport_code\": {\"description\": \"The name of Destination airport code\", \"type\": \"string\"}, \"departure_date\": {\"description\": \"The date of outbound flight\", \"type\": \"string\"}, \"return_date\": {\"description\": \"The date of return flight\", \"type\": \"string\"}}}, {\"name\": \"booking_hotels\", \"description\": \"booking hotel\", \"parameters\": {\"destination\": {\"description\": \"The name of the city\", \"type\": \"string\"}, \"check_in_date\": {\"description\": \"The date of check in\", \"type\": \"string\"}, \"checkout_date\": {\"description\": \"The date of check out\", \"type\": \"string\"}}}]";
string larkGrammar = "start: TEXT | fun_call\n" +
                     "TEXT: /[^{](.|\\n)*/\n" +
                     "fun_call: <|tool_call|> %json {\"anyOf\": " + functions + "} <|/tool_call|>";

using GeneratorParams generatorParams = new(model);
generatorParams.SetSearchOption("max_length", 4096);
generatorParams.SetSearchOption("temperature", 0.00001);
generatorParams.SetSearchOption("top_p", 1.0);
generatorParams.SetSearchOption("do_sample", false);
generatorParams.SetGuidance("lark_grammar", larkGrammar);
using Generator generator = new(model, generatorParams);

string systemMessageContent = "You are a helpful assistant with these tools.";
string userMessageContent =
    "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , then book hotel from 2025-12-04 to 2025-12-10 in Paris";

string messages = $"[{{\"role\": \"system\", \"content\": \"{systemMessageContent}\", \"tools\": {functions}}},{{\"role\": \"user\", \"content\": \"{userMessageContent}\"}}]";
string prompt = tokenizer.ApplyChatTemplate(null!, messages, null!, true);
ReadOnlySpan<int> tokens = tokenizer.Encode(prompt)[0];
generator.AppendTokens(tokens);

while (!generator.IsDone())
{
    generator.GenerateNextToken();
    int newToken = generator.GetSequence(0)[^1];
    string? decodedToken = tokenizerStream.Decode(newToken);
    Console.Write(decodedToken);
    if (newToken == 200025)
    {
        Console.WriteLine("[DEBUG: Tool call token detected!]");
    }

    if (newToken == 200026)
    {
        Console.WriteLine("\n[DEBUG: Tool call token detected!]");
    }
}