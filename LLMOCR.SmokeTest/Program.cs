using LLMOCR;

if (args.Length < 2)
{
    Console.WriteLine("Kullanim: dotnet run --project .\\LLMOCR.SmokeTest -- <girdiDosyasi> <ciktiTxt>");
    return;
}

using var processor = DocumentProcessor.FromEnvironment(".env");
var text = await processor.ExtractAndSave(args[0], args[1]);
Console.WriteLine($"Tamamlandi. Karakter sayisi: {text.Length}");
