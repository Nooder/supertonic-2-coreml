//
//  TTSViewModel.swift
//  supertonic2-coreml-ios-test
//
//  Created by Codex.
//

import Foundation
import Combine

@MainActor
final class TTSViewModel: ObservableObject {
    enum ModelLoadReason: String {
        case warmup
        case onDemand
        case settingsChange

        var displayName: String {
            switch self {
            case .warmup:
                return "Warm-up"
            case .onDemand:
                return "On-demand"
            case .settingsChange:
                return "Reload"
            }
        }

        var loadingMessage: String {
            switch self {
            case .warmup:
                return "Warming up Core ML models…"
            case .onDemand:
                return "Loading Core ML models…"
            case .settingsChange:
                return "Reloading Core ML models…"
            }
        }
    }

    struct Metrics {
        let audioSeconds: Double
        let elapsedSeconds: Double
        let timing: TTSService.Timing
        let memoryBeforeMB: Double?
        let memoryAfterMB: Double?

        var rtf: Double {
            guard audioSeconds > 0 else { return 0 }
            return elapsedSeconds / audioSeconds
        }
    }

    struct SamplePrompt: Identifiable {
        let id = UUID()
        let title: String
        let text: String
        let language: TTSService.Language
    }

    @Published var text: String = "This is Supertonic 2 running on Core ML with int8 weights, tuned for faster inference on older iPhones."
    @Published var selectedVoice: String = "F1"
    @Published var language: TTSService.Language = .en
    @Published var steps: Int = 20
    @Published var speed: Double = 1.05
    @Published var silenceSeconds: Double = 0.3
    @Published var computeUnits: TTSService.ComputeUnits = .all

    @Published var isGenerating: Bool = false
    @Published var isPlaying: Bool = false
    @Published var isLoadingModels: Bool = false
    @Published var errorMessage: String?
    @Published var metrics: Metrics?
    @Published var audioURL: URL?
    @Published var availableVoices: [String] = []
    @Published var modelLoadSeconds: Double?
    @Published var modelLoadReason: ModelLoadReason?
    @Published var modelLoadComputeUnits: TTSService.ComputeUnits?
    @Published var currentLoadReason: ModelLoadReason?

    let samples: [SamplePrompt] = [
        SamplePrompt(
            title: "Warm intro",
            text: "Hello. This is a quick Core ML quality check on an iPhone XR. We want steady latency and clear voice.",
            language: .en
        ),
        SamplePrompt(
            title: "Tech update",
            text: "We trimmed memory by compressing weights and kept the full pipeline in Core ML for better stability.",
            language: .en
        ),
        SamplePrompt(
            title: "Spanish sample",
            text: "Hola. Esta es una prueba breve para revisar claridad y ritmo en el dispositivo.",
            language: .es
        )
    ]

    private var service: TTSService?
    private var loadedComputeUnits: TTSService.ComputeUnits?
    private let player = AudioPlayer()
    private var pendingGenerate: Bool = false
    private var pendingModelLoad: Bool = false
    private var pendingLoadReason: ModelLoadReason = .onDemand

    func startup() {
        // Warm-start the app by loading voices and preloading models.
        loadVoices()
        warmUpModels()
    }

    func reloadModels() {
        requestModelLoad(reason: .settingsChange)
    }

    var loadingMessage: String {
        (currentLoadReason ?? .onDemand).loadingMessage
    }

    func loadVoices() {
        Task.detached {
            do {
                let voices = try TTSService.availableVoiceNames()
                await MainActor.run {
                    self.availableVoices = voices
                    if !voices.contains(self.selectedVoice) {
                        self.selectedVoice = voices.first ?? self.selectedVoice
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Failed to load voices: \(error.localizedDescription)"
                }
            }
        }
    }

    func generate() {
        // Ensure models are loaded for the current compute units before generating.
        guard let service = service, loadedComputeUnits == computeUnits else {
            pendingGenerate = true
            requestModelLoad(reason: .onDemand)
            return
        }
        isGenerating = true
        errorMessage = nil
        metrics = nil
        audioURL = nil

        let text = self.text
        let language = self.language
        let voice = self.selectedVoice
        let steps = self.steps
        let speed = self.speed
        let silence = self.silenceSeconds

        Task.detached {
            let start = Date()
            let memBefore = MemoryUsage.currentFootprintMB()
            do {
                // Run the full TTS pipeline off the main thread.
                let result = try service.synthesize(
                    text: text,
                    language: language,
                    voiceName: voice,
                    steps: steps,
                    speed: speed,
                    silenceSeconds: silence
                )
                let elapsed = Date().timeIntervalSince(start)
                let memAfter = MemoryUsage.currentFootprintMB()
                await MainActor.run {
                    self.audioURL = result.url
                    self.metrics = Metrics(
                        audioSeconds: result.audioSeconds,
                        elapsedSeconds: elapsed,
                        timing: result.timing,
                        memoryBeforeMB: memBefore,
                        memoryAfterMB: memAfter
                    )
                    self.isGenerating = false
                    self.play(url: result.url)
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isGenerating = false
                }
            }
        }
    }

    func togglePlay() {
        if isPlaying {
            player.stop()
            isPlaying = false
        } else if let url = audioURL {
            play(url: url)
        }
    }

    private func play(url: URL) {
        player.play(url: url) { [weak self] in
            DispatchQueue.main.async {
                self?.isPlaying = false
            }
        }
        isPlaying = true
    }

    private func warmUpModels() {
        Task { @MainActor in
            guard service == nil else { return }
            // Small delay allows the UI to settle before heavy model loads.
            try? await Task.sleep(nanoseconds: 250_000_000)
            requestModelLoad(reason: .warmup)
        }
    }

    private func requestModelLoad(reason: ModelLoadReason) {
        if isLoadingModels {
            // If the user changes settings during a load, queue a reload.
            if reason == .settingsChange {
                pendingModelLoad = true
                pendingLoadReason = reason
            }
            return
        }

        isLoadingModels = true
        currentLoadReason = reason
        errorMessage = nil
        let computeUnits = computeUnits
        let start = Date()

        Task.detached {
            do {
                // Model creation is expensive; keep it off the main thread.
                let service = try TTSService(computeUnits: computeUnits)
                let elapsed = Date().timeIntervalSince(start)
                await MainActor.run {
                    self.service = service
                    self.loadedComputeUnits = computeUnits
                    self.isLoadingModels = false
                    self.currentLoadReason = nil
                    self.modelLoadSeconds = elapsed
                    self.modelLoadReason = reason
                    self.modelLoadComputeUnits = computeUnits

                    // Apply any queued reloads and generation requests.
                    if self.pendingModelLoad {
                        self.pendingModelLoad = false
                        let pendingReason = self.pendingLoadReason
                        if pendingReason == .settingsChange && self.loadedComputeUnits != self.computeUnits {
                            self.requestModelLoad(reason: pendingReason)
                            return
                        }
                    }

                    if self.pendingGenerate {
                        self.pendingGenerate = false
                        self.generate()
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Failed to load models: \(error.localizedDescription)"
                    self.isLoadingModels = false
                    self.currentLoadReason = nil
                    self.pendingGenerate = false
                    if self.pendingModelLoad {
                        self.pendingModelLoad = false
                        let pendingReason = self.pendingLoadReason
                        self.requestModelLoad(reason: pendingReason)
                    }
                }
            }
        }
    }
}
