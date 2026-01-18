//
//  TTSService.swift
//  supertonic2-coreml-ios-test
//
//  Created by Codex.
//

import Foundation
import CoreML

final class TTSService {
    enum Language: String, CaseIterable, Identifiable {
        case en
        case ko
        case es
        case pt
        case fr

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .en: return "English"
            case .ko: return "Korean"
            case .es: return "Spanish"
            case .pt: return "Portuguese"
            case .fr: return "French"
            }
        }
    }

    enum ComputeUnits: String, CaseIterable, Identifiable {
        case cpu
        case cpuAndGPU
        case all

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .cpu: return "CPU"
            case .cpuAndGPU: return "CPU+GPU"
            case .all: return "All"
            }
        }

        var coreMLValue: MLComputeUnits {
            switch self {
            case .cpu: return .cpuOnly
            case .cpuAndGPU: return .cpuAndGPU
            case .all: return .all
            }
        }
    }

    struct Timing {
        var durationPredictor: Double = 0
        var textEncoder: Double = 0
        var vectorEstimator: Double = 0
        var vocoder: Double = 0

        var total: Double {
            durationPredictor + textEncoder + vectorEstimator + vocoder
        }
    }

    struct Result {
        let url: URL
        let audioSeconds: Double
        let timing: Timing
    }

    private struct Config: Decodable {
        let ttl: TTLConfig
        let ae: AEConfig

        enum CodingKeys: String, CodingKey {
            case ttl
            case ae
        }
    }

    private struct TTLConfig: Decodable {
        let chunk_compress_factor: Int
    }

    private struct AEConfig: Decodable {
        let sample_rate: Int
        let base_chunk_size: Int
    }

    private struct Embedding {
        let vocabSize: Int
        let dim: Int
        let weights: [Float]
    }

    private struct VoiceFile: Decodable {
        struct Tensor: Decodable {
            let data: [[[Float]]]
            let dims: [Int]
        }

        let style_ttl: Tensor
        let style_dp: Tensor

        enum CodingKeys: String, CodingKey {
            case style_ttl
            case style_dp
        }
    }

    private struct VoiceStyle {
        let ttl: MLMultiArray
        let dp: MLMultiArray
    }

    private enum TTSError: Error, LocalizedError {
        case missingResource(String)
        case invalidResource(String)
        case invalidText(String)
        case invalidModel(String)

        var errorDescription: String? {
            switch self {
            case .missingResource(let message): return message
            case .invalidResource(let message): return message
            case .invalidText(let message): return message
            case .invalidModel(let message): return message
            }
        }
    }

    private let dpModel: MLModel
    private let teModel: MLModel
    private let veModel: MLModel
    private let vocModel: MLModel

    private let unicodeIndexer: [Int]
    private let embeddingDP: Embedding
    private let embeddingTE: Embedding
    private let voiceDir: URL

    private let sampleRate: Int
    private let baseChunkSize: Int
    private let chunkCompressFactor: Int

    private let maxTextLen: Int
    private let latentDim: Int
    private let latentLenMax: Int

    private var voiceCache: [String: VoiceStyle] = [:]

    private static let compiledModelCacheLock = NSLock()
    private static var compiledModelCache: [URL: URL] = [:]

    init(computeUnits: ComputeUnits) throws {
        let resources = try Self.locateResources()
        let coremlDir = try? Self.locateSubdirectory("coreml_int8", in: resources)
        voiceDir = try Self.locateSubdirectory("voice_styles", in: resources)
        let onnxDir = try Self.locateSubdirectory("onnx", in: resources)
        let embeddingsDir = try Self.locateSubdirectory("embeddings", in: resources)

        let configURL = onnxDir.appendingPathComponent("tts.json")
        let cfg = try Self.loadConfig(configURL)
        sampleRate = cfg.ae.sample_rate
        baseChunkSize = cfg.ae.base_chunk_size
        chunkCompressFactor = cfg.ttl.chunk_compress_factor

        let indexerURL = onnxDir.appendingPathComponent("unicode_indexer.json")
        unicodeIndexer = try Self.loadIndexer(indexerURL)

        embeddingDP = try Self.loadEmbedding(
            dataURL: embeddingsDir.appendingPathComponent("char_embedder_dp.fp32.bin"),
            shapeURL: embeddingsDir.appendingPathComponent("char_embedder_dp.shape.json")
        )
        embeddingTE = try Self.loadEmbedding(
            dataURL: embeddingsDir.appendingPathComponent("char_embedder_te.fp32.bin"),
            shapeURL: embeddingsDir.appendingPathComponent("char_embedder_te.shape.json")
        )

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits.coreMLValue
        config.allowLowPrecisionAccumulationOnGPU = true

        dpModel = try Self.loadModel(named: "duration_predictor_mlprogram", coremlDir: coremlDir, configuration: config)
        teModel = try Self.loadModel(named: "text_encoder_mlprogram", coremlDir: coremlDir, configuration: config)
        veModel = try Self.loadModel(named: "vector_estimator_mlprogram", coremlDir: coremlDir, configuration: config)
        vocModel = try Self.loadModel(named: "vocoder_mlprogram", coremlDir: coremlDir, configuration: config)

        maxTextLen = try Self.extractShape(model: dpModel, inputName: "text_mask").last ?? 300
        let latentShape = try Self.extractShape(model: veModel, inputName: "noisy_latent")
        guard latentShape.count == 3 else {
            throw TTSError.invalidModel("Unexpected latent input shape: \(latentShape)")
        }
        latentDim = latentShape[1]
        latentLenMax = latentShape[2]
    }

    static func availableVoiceNames() throws -> [String] {
        let resources = try locateResources()
        let voiceDir = try locateSubdirectory("voice_styles", in: resources)
        let files = try FileManager.default.contentsOfDirectory(atPath: voiceDir.path)
        return files
            .filter { $0.lowercased().hasSuffix(".json") }
            .map { $0.replacingOccurrences(of: ".json", with: "") }
            .sorted()
    }

    func synthesize(
        text: String,
        language: Language,
        voiceName: String,
        steps: Int,
        speed: Double,
        silenceSeconds: Double
    ) throws -> Result {
        let voice = try loadVoiceStyle(named: voiceName)
        let maxLen = language == .ko ? min(maxTextLen, 120) : maxTextLen
        let chunks = chunkText(text, maxLen: maxLen)

        var timing = Timing()
        var fullAudio: [Float] = []

        for chunk in chunks {
            let processed = preprocessText(chunk, lang: language.rawValue)
            let (textIds, textMask) = try buildTextInputs(processedText: processed, maxLen: maxTextLen)
            let textEmbedDP = try buildTextEmbed(textIds: textIds, embedding: embeddingDP, maxLen: maxTextLen)
            let textEmbedTE = try buildTextEmbed(textIds: textIds, embedding: embeddingTE, maxLen: maxTextLen)

            let dpStart = CFAbsoluteTimeGetCurrent()
            let duration = try runDurationPredictor(styleDP: voice.dp, textMask: textMask, textEmbed: textEmbedDP)
            timing.durationPredictor += CFAbsoluteTimeGetCurrent() - dpStart

            let teStart = CFAbsoluteTimeGetCurrent()
            let textEmb = try runTextEncoder(styleTTL: voice.ttl, textMask: textMask, textEmbed: textEmbedTE)
            timing.textEncoder += CFAbsoluteTimeGetCurrent() - teStart

            let adjustedDuration = max(Double(duration) / max(speed, 0.01), 0.05)
            let maxDuration = maxDurationSeconds()
            let clippedDuration = min(adjustedDuration, maxDuration)

            let (noisyLatent, latentMask) = try sampleNoisyLatent(durationSeconds: clippedDuration)
            let veStart = CFAbsoluteTimeGetCurrent()
            let denoised = try runVectorEstimator(
                noisyLatent: noisyLatent,
                textEmb: textEmb,
                styleTTL: voice.ttl,
                latentMask: latentMask,
                textMask: textMask,
                steps: steps
            )
            timing.vectorEstimator += CFAbsoluteTimeGetCurrent() - veStart

            let vocStart = CFAbsoluteTimeGetCurrent()
            let wav = try runVocoder(latent: denoised)
            timing.vocoder += CFAbsoluteTimeGetCurrent() - vocStart

            let trimSamples = min(Int(Double(sampleRate) * clippedDuration), wav.count)
            if trimSamples > 0 {
                if !fullAudio.isEmpty {
                    fullAudio.append(contentsOf: Array(repeating: 0, count: Int(Double(sampleRate) * silenceSeconds)))
                }
                fullAudio.append(contentsOf: wav.prefix(trimSamples))
            }
        }

        normalizeAudio(&fullAudio)
        let tmpURL = FileManager.default.temporaryDirectory.appendingPathComponent("supertonic_tts_\(UUID().uuidString).wav")
        try writeWavFile(tmpURL.path, fullAudio, sampleRate)
        let audioSeconds = Double(fullAudio.count) / Double(sampleRate)
        return Result(url: tmpURL, audioSeconds: audioSeconds, timing: timing)
    }

    // MARK: - Model Runners

    private func runDurationPredictor(styleDP: MLMultiArray, textMask: MLMultiArray, textEmbed: MLMultiArray) throws -> Float {
        let inputs: [String: MLMultiArray] = [
            "style_dp": styleDP,
            "text_mask": textMask,
            "text_embed": textEmbed
        ]
        let output = try predict(model: dpModel, inputs: inputs)
        guard let duration = output["duration"] else {
            throw TTSError.invalidModel("Missing duration output.")
        }
        return readScalar(duration)
    }

    private func runTextEncoder(styleTTL: MLMultiArray, textMask: MLMultiArray, textEmbed: MLMultiArray) throws -> MLMultiArray {
        let inputs: [String: MLMultiArray] = [
            "style_ttl": styleTTL,
            "text_mask": textMask,
            "text_embed": textEmbed
        ]
        let output = try predict(model: teModel, inputs: inputs)
        guard let textEmb = output["text_emb"] else {
            throw TTSError.invalidModel("Missing text_emb output.")
        }
        return textEmb
    }

    private func runVectorEstimator(
        noisyLatent: MLMultiArray,
        textEmb: MLMultiArray,
        styleTTL: MLMultiArray,
        latentMask: MLMultiArray,
        textMask: MLMultiArray,
        steps: Int
    ) throws -> MLMultiArray {
        var latent = noisyLatent
        let totalStep = try makeScalar(Float(steps))
        for step in 0..<steps {
            let currentStep = try makeScalar(Float(step))
            let inputs: [String: MLMultiArray] = [
                "noisy_latent": latent,
                "text_emb": textEmb,
                "style_ttl": styleTTL,
                "latent_mask": latentMask,
                "text_mask": textMask,
                "current_step": currentStep,
                "total_step": totalStep
            ]
            let output = try predict(model: veModel, inputs: inputs)
            guard let denoised = output["denoised_latent"] else {
                throw TTSError.invalidModel("Missing denoised_latent output.")
            }
            latent = denoised
        }
        return latent
    }

    private func runVocoder(latent: MLMultiArray) throws -> [Float] {
        let output = try predict(model: vocModel, inputs: ["latent": latent])
        guard let wav = output["wav_tts"] else {
            throw TTSError.invalidModel("Missing wav_tts output.")
        }
        return toFloatArray(wav)
    }

    // MARK: - Text Processing

    private func preprocessText(_ text: String, lang: String) -> String {
        var text = text.decomposedStringWithCompatibilityMapping
        text = text.unicodeScalars.filter { scalar in
            let value = scalar.value
            return !((value >= 0x1F600 && value <= 0x1F64F) ||
                     (value >= 0x1F300 && value <= 0x1F5FF) ||
                     (value >= 0x1F680 && value <= 0x1F6FF) ||
                     (value >= 0x1F700 && value <= 0x1F77F) ||
                     (value >= 0x1F780 && value <= 0x1F7FF) ||
                     (value >= 0x1F800 && value <= 0x1F8FF) ||
                     (value >= 0x1F900 && value <= 0x1F9FF) ||
                     (value >= 0x1FA00 && value <= 0x1FA6F) ||
                     (value >= 0x1FA70 && value <= 0x1FAFF) ||
                     (value >= 0x2600 && value <= 0x26FF) ||
                     (value >= 0x2700 && value <= 0x27BF) ||
                     (value >= 0x1F1E6 && value <= 0x1F1FF))
        }.map { String($0) }.joined()

        let replacements: [String: String] = [
            "–": "-",
            "‑": "-",
            "—": "-",
            "_": " ",
            "\u{201C}": "\"",
            "\u{201D}": "\"",
            "\u{2018}": "'",
            "\u{2019}": "'",
            "´": "'",
            "`": "'",
            "[": " ",
            "]": " ",
            "|": " ",
            "/": " ",
            "#": " ",
            "→": " ",
            "←": " "
        ]
        for (old, new) in replacements {
            text = text.replacingOccurrences(of: old, with: new)
        }

        for symbol in ["♥", "☆", "♡", "©", "\\"] {
            text = text.replacingOccurrences(of: symbol, with: "")
        }

        let exprReplacements: [String: String] = [
            "@": " at ",
            "e.g.,": "for example, ",
            "i.e.,": "that is, "
        ]
        for (old, new) in exprReplacements {
            text = text.replacingOccurrences(of: old, with: new)
        }

        text = text.replacingOccurrences(of: " ,", with: ",")
        text = text.replacingOccurrences(of: " .", with: ".")
        text = text.replacingOccurrences(of: " !", with: "!")
        text = text.replacingOccurrences(of: " ?", with: "?")
        text = text.replacingOccurrences(of: " ;", with: ";")
        text = text.replacingOccurrences(of: " :", with: ":")
        text = text.replacingOccurrences(of: " '", with: "'")

        while text.contains("\"\"") { text = text.replacingOccurrences(of: "\"\"", with: "\"") }
        while text.contains("''") { text = text.replacingOccurrences(of: "''", with: "'") }
        while text.contains("``") { text = text.replacingOccurrences(of: "``", with: "`") }

        let whitespacePattern = try? NSRegularExpression(pattern: "\\s+")
        let whitespaceRange = NSRange(text.startIndex..., in: text)
        if let whitespacePattern = whitespacePattern {
            text = whitespacePattern.stringByReplacingMatches(in: text, range: whitespaceRange, withTemplate: " ")
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        if !text.isEmpty {
            let punctPattern = try? NSRegularExpression(pattern: "[.!?;:,'\"\\u201C\\u201D\\u2018\\u2019)\\]}…。」』】〉》›»]$")
            let punctRange = NSRange(text.startIndex..., in: text)
            if punctPattern?.firstMatch(in: text, range: punctRange) == nil {
                text += "."
            }
        }

        return "<\(lang)>\(text)</\(lang)>"
    }

    private func chunkText(_ text: String, maxLen: Int) -> [String] {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedText.isEmpty { return [""] }

        let paragraphPattern = try? NSRegularExpression(pattern: "\\n\\s*\\n")
        let paraRange = NSRange(trimmedText.startIndex..., in: trimmedText)
        var paragraphs: [String] = []
        var lastEnd = trimmedText.startIndex
        paragraphPattern?.enumerateMatches(in: trimmedText, range: paraRange) { match, _, _ in
            guard let match = match, let range = Range(match.range, in: trimmedText) else { return }
            paragraphs.append(String(trimmedText[lastEnd..<range.lowerBound]))
            lastEnd = range.upperBound
        }
        if lastEnd < trimmedText.endIndex {
            paragraphs.append(String(trimmedText[lastEnd...]))
        }
        if paragraphs.isEmpty {
            paragraphs = [trimmedText]
        }

        var chunks: [String] = []
        for para in paragraphs {
            let trimmed = para.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            if trimmed.count <= maxLen {
                chunks.append(trimmed)
                continue
            }

            let sentences = splitSentences(trimmed)
            var current = ""
            var currentLen = 0
            for sentence in sentences {
                let trimmedSentence = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedSentence.isEmpty { continue }
                let sentenceLen = trimmedSentence.count

                if sentenceLen > maxLen {
                    if !current.isEmpty {
                        chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                        current = ""
                        currentLen = 0
                    }
                    let parts = trimmedSentence.components(separatedBy: ",")
                    for part in parts {
                        let trimmedPart = part.trimmingCharacters(in: .whitespacesAndNewlines)
                        if trimmedPart.isEmpty { continue }
                        let partLen = trimmedPart.count
                        if partLen > maxLen {
                            let words = trimmedPart.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
                            var wordChunk = ""
                            var wordLen = 0
                            for word in words {
                                let wlen = word.count
                                if wordLen + wlen + 1 > maxLen && !wordChunk.isEmpty {
                                    chunks.append(wordChunk.trimmingCharacters(in: .whitespacesAndNewlines))
                                    wordChunk = ""
                                    wordLen = 0
                                }
                                if !wordChunk.isEmpty {
                                    wordChunk += " "
                                    wordLen += 1
                                }
                                wordChunk += word
                                wordLen += wlen
                            }
                            if !wordChunk.isEmpty {
                                chunks.append(wordChunk.trimmingCharacters(in: .whitespacesAndNewlines))
                            }
                        } else {
                            if currentLen + partLen + 1 > maxLen && !current.isEmpty {
                                chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                                current = ""
                                currentLen = 0
                            }
                            if !current.isEmpty {
                                current += ", "
                                currentLen += 2
                            }
                            current += trimmedPart
                            currentLen += partLen
                        }
                    }
                    continue
                }

                if currentLen + sentenceLen + 1 > maxLen && !current.isEmpty {
                    chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                    current = ""
                    currentLen = 0
                }
                if !current.isEmpty {
                    current += " "
                    currentLen += 1
                }
                current += trimmedSentence
                currentLen += sentenceLen
            }
            if !current.isEmpty {
                chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        return chunks.isEmpty ? [""] : chunks
    }

    private func splitSentences(_ text: String) -> [String] {
        let abbreviations: Set<String> = [
            "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
            "St.", "Ave.", "Rd.", "Blvd.", "Dept.", "Inc.", "Ltd.",
            "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D."
        ]
        let regex = try? NSRegularExpression(pattern: "([.!?])\\s+")
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex?.matches(in: text, range: range) ?? []
        if matches.isEmpty { return [text] }

        var sentences: [String] = []
        var lastEnd = text.startIndex
        for match in matches {
            guard let matchRange = Range(match.range, in: text) else { continue }
            let beforePunc = String(text[lastEnd..<matchRange.lowerBound])
            let puncRange = Range(NSRange(location: match.range.location, length: 1), in: text)!
            let punc = String(text[puncRange])
            let combined = beforePunc.trimmingCharacters(in: .whitespaces) + punc
            let isAbbrev = abbreviations.contains(where: { combined.hasSuffix($0) })
            if !isAbbrev {
                sentences.append(String(text[lastEnd..<matchRange.upperBound]))
                lastEnd = matchRange.upperBound
            }
        }
        if lastEnd < text.endIndex {
            sentences.append(String(text[lastEnd...]))
        }
        return sentences.isEmpty ? [text] : sentences
    }

    private func buildTextInputs(processedText: String, maxLen: Int) throws -> ([Int], MLMultiArray) {
        let scalars = processedText.unicodeScalars
        var ids: [Int] = []
        ids.reserveCapacity(scalars.count)
        for scalar in scalars {
            let value = Int(scalar.value)
            guard value < unicodeIndexer.count else {
                throw TTSError.invalidText("Unsupported character in text.")
            }
            let idx = unicodeIndexer[value]
            guard idx >= 0 else {
                throw TTSError.invalidText("Unsupported character in text.")
            }
            ids.append(idx)
        }
        if ids.count > maxLen {
            throw TTSError.invalidText("Text length \(ids.count) exceeds max length \(maxLen).")
        }
        let mask = try makeMask(length: ids.count, maxLen: maxLen)
        return (ids, mask)
    }

    private func buildTextEmbed(textIds: [Int], embedding: Embedding, maxLen: Int) throws -> MLMultiArray {
        let shape = [1, embedding.dim, maxLen]
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        let strides = array.strides.map { Int(truncating: $0) }
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)

        for t in 0..<maxLen {
            if t < textIds.count {
                let id = textIds[t]
                if id < 0 || id >= embedding.vocabSize {
                    throw TTSError.invalidText("Text id out of range: \(id)")
                }
                let base = id * embedding.dim
                for d in 0..<embedding.dim {
                    let offset = d * strides[1] + t * strides[2]
                    ptr[offset] = embedding.weights[base + d]
                }
            } else {
                for d in 0..<embedding.dim {
                    let offset = d * strides[1] + t * strides[2]
                    ptr[offset] = 0
                }
            }
        }
        return array
    }

    // MARK: - Latent Sampling

    private func sampleNoisyLatent(durationSeconds: Double) throws -> (MLMultiArray, MLMultiArray) {
        let wavLen = Int(durationSeconds * Double(sampleRate))
        let chunkSize = baseChunkSize * chunkCompressFactor
        let latentLen = min((wavLen + chunkSize - 1) / chunkSize, latentLenMax)

        let latentMask = try makeMask(length: latentLen, maxLen: latentLenMax)
        let noisyLatent = try MLMultiArray(shape: [1, latentDim, latentLenMax].map { NSNumber(value: $0) }, dataType: .float32)
        let strides = noisyLatent.strides.map { Int(truncating: $0) }
        let ptr = noisyLatent.dataPointer.bindMemory(to: Float32.self, capacity: noisyLatent.count)

        for d in 0..<latentDim {
            for t in 0..<latentLenMax {
                let offset = d * strides[1] + t * strides[2]
                if t < latentLen {
                    ptr[offset] = randomNormal()
                } else {
                    ptr[offset] = 0
                }
            }
        }
        return (noisyLatent, latentMask)
    }

    private func randomNormal() -> Float {
        let u1 = max(Float.random(in: 0..<1), 1e-6)
        let u2 = Float.random(in: 0..<1)
        return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
    }

    private func makeMask(length: Int, maxLen: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 1, maxLen].map { NSNumber(value: $0) }, dataType: .float32)
        let strides = array.strides.map { Int(truncating: $0) }
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        for t in 0..<maxLen {
            let offset = t * strides[2]
            ptr[offset] = t < length ? 1 : 0
        }
        return array
    }

    private func makeScalar(_ value: Float) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: 1)
        ptr[0] = value
        return array
    }

    // MARK: - CoreML Helpers

    private func predict(model: MLModel, inputs: [String: MLMultiArray]) throws -> [String: MLMultiArray] {
        let featureInputs: [String: MLFeatureValue] = inputs.mapValues { MLFeatureValue(multiArray: $0) }
        let provider = try MLDictionaryFeatureProvider(dictionary: featureInputs)
        let output = try model.prediction(from: provider)
        var result: [String: MLMultiArray] = [:]
        for name in output.featureNames {
            if let value = output.featureValue(for: name)?.multiArrayValue {
                result[name] = value
            }
        }
        return result
    }

    private func readScalar(_ array: MLMultiArray) -> Float {
        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
            return Float(ptr[0])
        }
        if array.dataType == .float16 {
            let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
            return Float(ptr[0])
        }
        if array.dataType == .double {
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
            return Float(ptr[0])
        }
        if array.dataType == .int32 {
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
            return Float(ptr[0])
        }
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        return Float(ptr[0])
    }

    private func toFloatArray(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count)).map { Float($0) }
        }
        if array.dataType == .float16 {
            let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count)).map { Float($0) }
        }
        if array.dataType == .double {
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count)).map { Float($0) }
        }
        if array.dataType == .int32 {
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count)).map { Float($0) }
        }
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count)).map { Float($0) }
    }

    // MARK: - Audio Output

    private func normalizeAudio(_ audio: inout [Float]) {
        guard !audio.isEmpty else { return }
        let maxAbs = audio.map { abs($0) }.max() ?? 0
        guard maxAbs > 1e-6 else { return }
        let scale = 0.95 / maxAbs
        for i in 0..<audio.count {
            audio[i] = audio[i] * Float(scale)
        }
    }

    private func writeWavFile(_ filename: String, _ audioData: [Float], _ sampleRate: Int) throws {
        let url = URL(fileURLWithPath: filename)
        let int16Data = audioData.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * 32767.0)
        }
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample) / 8
        let blockAlign = numChannels * bitsPerSample / 8
        let dataSize = UInt32(int16Data.count * 2)

        var data = Data()
        data.append("RIFF".data(using: .ascii)!)
        withUnsafeBytes(of: UInt32(36 + dataSize).littleEndian) { data.append(contentsOf: $0) }
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        withUnsafeBytes(of: UInt32(16).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(1).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: numChannels.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: byteRate.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: blockAlign.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: bitsPerSample.littleEndian) { data.append(contentsOf: $0) }
        data.append("data".data(using: .ascii)!)
        withUnsafeBytes(of: dataSize.littleEndian) { data.append(contentsOf: $0) }
        int16Data.withUnsafeBytes { data.append(contentsOf: $0) }

        try data.write(to: url)
    }

    // MARK: - Resource Loading

    private func loadVoiceStyle(named name: String) throws -> VoiceStyle {
        if let cached = voiceCache[name] {
            return cached
        }
        let url = voiceDir.appendingPathComponent("\(name).json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw TTSError.missingResource("Missing voice style: \(name)")
        }
        let data = try Data(contentsOf: url)
        let voice = try JSONDecoder().decode(VoiceFile.self, from: data)
        let ttlFlat = flatten(voice.style_ttl.data, reserve: voice.style_ttl.dims.reduce(1, *))
        let dpFlat = flatten(voice.style_dp.data, reserve: voice.style_dp.dims.reduce(1, *))
        let ttl = try makeMultiArray(shape: voice.style_ttl.dims, data: ttlFlat)
        let dp = try makeMultiArray(shape: voice.style_dp.dims, data: dpFlat)
        let style = VoiceStyle(ttl: ttl, dp: dp)
        voiceCache[name] = style
        return style
    }

    private func flatten(_ data: [[[Float]]], reserve: Int) -> [Float] {
        var flat: [Float] = []
        flat.reserveCapacity(reserve)
        for a in data {
            for b in a {
                flat.append(contentsOf: b)
            }
        }
        return flat
    }

    private func makeMultiArray(shape: [Int], data: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        guard array.count == data.count else {
            throw TTSError.invalidResource("Shape \(shape) does not match data count \(data.count).")
        }
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        data.withUnsafeBufferPointer { buffer in
            if let base = buffer.baseAddress {
                ptr.update(from: base, count: buffer.count)
            }
        }
        return array
    }

    private func maxDurationSeconds() -> Double {
        let chunkSize = baseChunkSize * chunkCompressFactor
        let maxSamples = latentLenMax * chunkSize
        return Double(maxSamples) / Double(sampleRate)
    }

    private static func locateResources() throws -> URL {
        guard let resourceURL = Bundle.main.resourceURL else {
            throw TTSError.missingResource("Bundle resource URL not found.")
        }
        return resourceURL
    }

    private static func locateSubdirectory(_ name: String, in resources: URL) throws -> URL {
        let fm = FileManager.default
        let candidates: [URL] = [
            resources.appendingPathComponent("SupertonicResources/\(name)", isDirectory: true),
            resources.appendingPathComponent("Resources/\(name)", isDirectory: true),
            resources.appendingPathComponent(name, isDirectory: true),
            resources.appendingPathComponent(name),
        ]
        for url in candidates where fm.fileExists(atPath: url.path) {
            return url
        }
        throw TTSError.missingResource("Missing resource folder: \(name). Ensure SupertonicResources/\(name) is included in the app bundle.")
    }

    private static func loadModel(
        named name: String,
        coremlDir: URL?,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        let modelURL = try locateModelURL(named: name, coremlDir: coremlDir)
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    private static func locateModelURL(named name: String, coremlDir: URL?) throws -> URL {
        if let bundled = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return bundled
        }

        if let coremlDir {
            let compiled = coremlDir.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiled.path) {
                return compiled
            }

            let package = coremlDir.appendingPathComponent("\(name).mlpackage")
            if FileManager.default.fileExists(atPath: package.path) {
                return try compileModel(at: package)
            }

            let model = coremlDir.appendingPathComponent("\(name).mlmodel")
            if FileManager.default.fileExists(atPath: model.path) {
                return try compileModel(at: model)
            }
        }

        throw TTSError.missingResource("Missing CoreML model: \(name). Ensure the compiled .mlmodelc is in the app bundle.")
    }

    private static func compileModel(at url: URL) throws -> URL {
        compiledModelCacheLock.lock()
        if let cached = compiledModelCache[url] {
            compiledModelCacheLock.unlock()
            return cached
        }
        compiledModelCacheLock.unlock()

        let compiled = try MLModel.compileModel(at: url)

        compiledModelCacheLock.lock()
        compiledModelCache[url] = compiled
        compiledModelCacheLock.unlock()
        return compiled
    }

    private static func loadConfig(_ url: URL) throws -> Config {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Config.self, from: data)
    }

    private static func loadIndexer(_ url: URL) throws -> [Int] {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data)
        guard let array = json as? [Int] else {
            throw TTSError.invalidResource("unicode_indexer.json is not an int array.")
        }
        return array
    }

    private static func loadEmbedding(dataURL: URL, shapeURL: URL) throws -> Embedding {
        let shapeData = try Data(contentsOf: shapeURL)
        let shapeJson = try JSONSerialization.jsonObject(with: shapeData) as? [String: Any]
        guard let shape = shapeJson?["shape"] as? [Int], shape.count == 2 else {
            throw TTSError.invalidResource("Embedding shape missing or invalid.")
        }
        let data = try Data(contentsOf: dataURL)
        let count = data.count / MemoryLayout<Float>.size
        var weights = [Float](repeating: 0, count: count)
        weights.withUnsafeMutableBytes { dest in
            dest.copyBytes(from: data)
        }
        let expected = shape[0] * shape[1]
        guard expected == weights.count else {
            throw TTSError.invalidResource("Embedding size mismatch: expected \(expected) floats, got \(weights.count).")
        }
        return Embedding(vocabSize: shape[0], dim: shape[1], weights: weights)
    }

    private static func extractShape(model: MLModel, inputName: String) throws -> [Int] {
        guard let desc = model.modelDescription.inputDescriptionsByName[inputName],
              let constraint = desc.multiArrayConstraint else {
            throw TTSError.invalidModel("Missing input constraint for \(inputName)")
        }
        return constraint.shape.map { Int(truncating: $0) }
    }
}
