using ImageMagick;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace LLMOCR;

public sealed class DocumentProcessor : IDisposable
{
    private readonly string _apiKey;
    private readonly string _defaultModel;
    private readonly Uri _chatCompletionsEndpoint;
    private readonly HttpClient _httpClient;
    private readonly bool _disposeHttpClient;

    public DocumentProcessor(string apiKey, HttpClient? httpClient = null)
        : this(new DocumentProcessorOptions { ApiKey = apiKey }, httpClient)
    {
    }

    public DocumentProcessor(DocumentProcessorOptions options, HttpClient? httpClient = null)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (string.IsNullOrWhiteSpace(options.ApiKey))
        {
            throw new ArgumentException("API key cannot be null or empty.", nameof(options.ApiKey));
        }

        var baseUrl = string.IsNullOrWhiteSpace(options.ApiBaseUrl)
            ? DocumentProcessorOptions.DefaultApiBaseUrl
            : options.ApiBaseUrl.Trim();

        _apiKey = options.ApiKey.Trim();
        _defaultModel = string.IsNullOrWhiteSpace(options.ModelName)
            ? DocumentProcessorOptions.DefaultModelName
            : options.ModelName.Trim();
        _chatCompletionsEndpoint = BuildChatCompletionsEndpoint(baseUrl);
        _httpClient = httpClient ?? new HttpClient();
        _disposeHttpClient = httpClient is null;
    }

    public static DocumentProcessor FromEnvironment(string envFilePath = ".env", HttpClient? httpClient = null)
    {
        var options = DocumentProcessorOptions.FromEnvironment(envFilePath);
        return new DocumentProcessor(options, httpClient);
    }

    public Task<IReadOnlyList<byte[]>> ConvertPdfToImages(
        string pdfPath,
        int density = 300,
        int quality = 92,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(pdfPath))
        {
            throw new ArgumentException("PDF path cannot be null or empty.", nameof(pdfPath));
        }

        if (!File.Exists(pdfPath))
        {
            throw new FileNotFoundException("PDF file was not found.", pdfPath);
        }

        if (!string.Equals(Path.GetExtension(pdfPath), ".pdf", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException("ConvertPdfToImages only accepts .pdf files.");
        }

        if (density <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(density), "Density must be greater than zero.");
        }

        if (quality is < 1 or > 100)
        {
            throw new ArgumentOutOfRangeException(nameof(quality), "Quality must be between 1 and 100.");
        }

        return Task.Run<IReadOnlyList<byte[]>>(() =>
        {
            var imagesAsBytes = new List<byte[]>();
            var readSettings = new MagickReadSettings
            {
                Density = new Density(density, density)
            };

            using var pages = new MagickImageCollection();
            try
            {
                pages.Read(pdfPath, readSettings);
            }
            catch (MagickDelegateErrorException ex) when (IsGhostscriptMissing(ex))
            {
                throw new InvalidOperationException(
                    "PDF processing requires Ghostscript. Install Ghostscript and make sure 'gswin64c.exe' is in PATH.",
                    ex);
            }

            foreach (var page in pages)
            {
                cancellationToken.ThrowIfCancellationRequested();
                page.Format = MagickFormat.Jpeg;
                page.Quality = (uint)quality;
                imagesAsBytes.Add(page.ToByteArray());
            }

            return imagesAsBytes;
        }, cancellationToken);
    }

    public async Task<string> ProcessImageWithLLM(
        byte[] imageBytes,
        string? model = null,
        string prompt = "Extract all readable text from this image. Return plain text only.",
        int maxTokens = 4000,
        CancellationToken cancellationToken = default)
    {
        if (imageBytes is null || imageBytes.Length == 0)
        {
            throw new ArgumentException("Image bytes cannot be null or empty.", nameof(imageBytes));
        }

        var modelToUse = string.IsNullOrWhiteSpace(model) ? _defaultModel : model.Trim();

        var normalizedImageBytes = NormalizeToJpeg(imageBytes);
        var base64Image = Convert.ToBase64String(normalizedImageBytes);
        var requestPayload = new
        {
            model = modelToUse,
            messages = new object[]
            {
                new
                {
                    role = "user",
                    content = new object[]
                    {
                        new { type = "text", text = prompt },
                        new
                        {
                            type = "image_url",
                            image_url = new
                            {
                                url = $"data:image/jpeg;base64,{base64Image}"
                            }
                        }
                    }
                }
            },
            max_tokens = maxTokens
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, _chatCompletionsEndpoint)
        {
            Content = new StringContent(
                JsonSerializer.Serialize(requestPayload),
                Encoding.UTF8,
                "application/json")
        };
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        var responseContent = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"LLM API request failed with status {(int)response.StatusCode} ({response.StatusCode}). " +
                $"Body: {Truncate(responseContent, 600)}");
        }

        using var json = JsonDocument.Parse(responseContent);
        if (TryExtractAssistantText(json, out var text))
        {
            return text;
        }

        throw new InvalidOperationException("LLM response did not contain extractable text.");
    }

    public async Task<string> ExtractAndSave(
        string inputPath,
        string outputTxtPath,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(inputPath))
        {
            throw new ArgumentException("Input path cannot be null or empty.", nameof(inputPath));
        }

        if (string.IsNullOrWhiteSpace(outputTxtPath))
        {
            throw new ArgumentException("Output path cannot be null or empty.", nameof(outputTxtPath));
        }

        if (!File.Exists(inputPath))
        {
            throw new FileNotFoundException("Input file was not found.", inputPath);
        }

        try
        {
            var extension = Path.GetExtension(inputPath).ToLowerInvariant();
            var images = new List<byte[]>();

            switch (extension)
            {
                case ".pdf":
                    images.AddRange(await ConvertPdfToImages(inputPath, cancellationToken: cancellationToken).ConfigureAwait(false));
                    break;
                case ".jpg":
                case ".jpeg":
                case ".png":
                    images.Add(await File.ReadAllBytesAsync(inputPath, cancellationToken).ConfigureAwait(false));
                    break;
                default:
                    throw new NotSupportedException("Only PDF, JPG, JPEG, and PNG files are supported.");
            }

            if (images.Count == 0)
            {
                throw new InvalidOperationException("No image content was produced from the input file.");
            }

            var resultBuilder = new StringBuilder();
            for (var i = 0; i < images.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var extractedText = await ProcessImageWithLLM(images[i], cancellationToken: cancellationToken).ConfigureAwait(false);

                if (images.Count > 1)
                {
                    resultBuilder.AppendLine($"--- Page {i + 1} ---");
                }

                resultBuilder.AppendLine(extractedText.Trim());
                if (i < images.Count - 1)
                {
                    resultBuilder.AppendLine();
                }
            }

            var finalText = resultBuilder.ToString().Trim();
            var outputDirectory = Path.GetDirectoryName(outputTxtPath);
            if (!string.IsNullOrWhiteSpace(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            await File.WriteAllTextAsync(outputTxtPath, finalText, Encoding.UTF8, cancellationToken).ConfigureAwait(false);
            return finalText;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to process and extract text from '{inputPath}'.", ex);
        }
    }

    public void Dispose()
    {
        if (_disposeHttpClient)
        {
            _httpClient.Dispose();
        }
    }

    private static Uri BuildChatCompletionsEndpoint(string apiBaseUrl)
    {
        if (string.IsNullOrWhiteSpace(apiBaseUrl))
        {
            throw new ArgumentException("API base URL cannot be null or empty.", nameof(apiBaseUrl));
        }

        var trimmed = apiBaseUrl.Trim().TrimEnd('/');
        if (!Uri.TryCreate(trimmed, UriKind.Absolute, out var baseUri))
        {
            throw new ArgumentException("API base URL is not a valid absolute URL.", nameof(apiBaseUrl));
        }

        if (trimmed.EndsWith("/chat/completions", StringComparison.OrdinalIgnoreCase))
        {
            return baseUri;
        }

        return new Uri($"{trimmed}/chat/completions", UriKind.Absolute);
    }

    private static bool TryExtractAssistantText(JsonDocument json, out string text)
    {
        text = string.Empty;

        if (!json.RootElement.TryGetProperty("choices", out var choices) || choices.GetArrayLength() == 0)
        {
            return false;
        }

        var firstChoice = choices[0];
        if (!firstChoice.TryGetProperty("message", out var message))
        {
            return false;
        }

        if (!message.TryGetProperty("content", out var content))
        {
            return false;
        }

        if (content.ValueKind == JsonValueKind.String)
        {
            text = content.GetString()?.Trim() ?? string.Empty;
            return !string.IsNullOrWhiteSpace(text);
        }

        if (content.ValueKind != JsonValueKind.Array)
        {
            return false;
        }

        var builder = new StringBuilder();
        foreach (var item in content.EnumerateArray())
        {
            if (!item.TryGetProperty("text", out var textElement) || textElement.ValueKind != JsonValueKind.String)
            {
                continue;
            }

            var piece = textElement.GetString();
            if (string.IsNullOrWhiteSpace(piece))
            {
                continue;
            }

            if (builder.Length > 0)
            {
                builder.AppendLine();
            }

            builder.Append(piece.Trim());
        }

        if (builder.Length == 0)
        {
            return false;
        }

        text = builder.ToString();
        return true;
    }

    private static string Truncate(string value, int maxLength)
    {
        if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
        {
            return value;
        }

        return value[..maxLength] + "...";
    }

    private static byte[] NormalizeToJpeg(byte[] imageBytes)
    {
        using var image = new MagickImage(imageBytes);
        image.Format = MagickFormat.Jpeg;
        image.Quality = 92;
        return image.ToByteArray();
    }

    private static bool IsGhostscriptMissing(MagickDelegateErrorException ex)
    {
        return ex.Message.Contains("gswin64c.exe", StringComparison.OrdinalIgnoreCase);
    }
}

public sealed class DocumentProcessorOptions
{
    public const string ApiKeyVariable = "LLMOCR_API_KEY";
    public const string BaseUrlVariable = "LLMOCR_BASE_URL";
    public const string ModelVariable = "LLMOCR_MODEL";
    public const string DefaultApiBaseUrl = "https://api.openai.com/v1";
    public const string DefaultModelName = "gpt-4o-mini";

    public string ApiKey { get; init; } = string.Empty;
    public string ApiBaseUrl { get; init; } = DefaultApiBaseUrl;
    public string ModelName { get; init; } = DefaultModelName;

    public static DocumentProcessorOptions FromEnvironment(string envFilePath = ".env")
    {
        var fileValues = LoadEnvFile(envFilePath);

        var apiKey = ReadSetting(ApiKeyVariable, fileValues, required: true)!;
        var apiBaseUrl = ReadSetting(BaseUrlVariable, fileValues, required: false) ?? DefaultApiBaseUrl;
        var modelName = ReadSetting(ModelVariable, fileValues, required: false) ?? DefaultModelName;

        return new DocumentProcessorOptions
        {
            ApiKey = apiKey,
            ApiBaseUrl = apiBaseUrl,
            ModelName = modelName
        };
    }

    private static string? ReadSetting(
        string key,
        IReadOnlyDictionary<string, string> fileValues,
        bool required)
    {
        var environmentValue = Environment.GetEnvironmentVariable(key);
        if (!string.IsNullOrWhiteSpace(environmentValue))
        {
            return environmentValue.Trim();
        }

        if (fileValues.TryGetValue(key, out var fileValue) && !string.IsNullOrWhiteSpace(fileValue))
        {
            return fileValue.Trim();
        }

        if (required)
        {
            throw new InvalidOperationException(
                $"Missing required setting '{key}'. Set it in OS environment variables or .env file.");
        }

        return null;
    }

    private static IReadOnlyDictionary<string, string> LoadEnvFile(string envFilePath)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (string.IsNullOrWhiteSpace(envFilePath))
        {
            return values;
        }

        var fullPath = Path.GetFullPath(envFilePath);
        if (!File.Exists(fullPath))
        {
            return values;
        }

        foreach (var rawLine in File.ReadLines(fullPath))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }

            var separatorIndex = line.IndexOf('=');
            if (separatorIndex <= 0)
            {
                continue;
            }

            var key = line[..separatorIndex].Trim();
            var value = line[(separatorIndex + 1)..].Trim();

            if (key.StartsWith("export ", StringComparison.OrdinalIgnoreCase))
            {
                key = key[7..].Trim();
            }

            if (key.Length == 0)
            {
                continue;
            }

            if ((value.StartsWith('"') && value.EndsWith('"')) ||
                (value.StartsWith('\'') && value.EndsWith('\'')))
            {
                value = value[1..^1];
            }

            values[key] = value;
        }

        return values;
    }
}
