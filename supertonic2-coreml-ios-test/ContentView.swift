//
//  ContentView.swift
//  supertonic2-coreml-ios-test
//
//  Created by Nader Beyzaei on 2026-01-16.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = TTSViewModel()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header

                    if viewModel.isLoadingModels {
                        ProgressView(viewModel.loadingMessage)
                    }

                    inputSection
                    controlSection
                    actionSection
                    metricsSection
                    samplesSection

                    if let error = viewModel.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.footnote)
                    }
                }
                .padding()
            }
            .navigationTitle("Supertonic2 CoreML")
            .onAppear { viewModel.startup() }
            .onChange(of: viewModel.computeUnits) { _ in
                viewModel.reloadModels()
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Int8 CoreML pipeline")
                .font(.headline)
            Text("iOS 15+ • duration + text encoder + vector estimator + vocoder")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Text")
                .font(.subheadline)
            TextEditor(text: $viewModel.text)
                .frame(minHeight: 140)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.4))
                )
        }
    }

    private var controlSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Voice")
                        .font(.subheadline)
                    Picker("Voice", selection: $viewModel.selectedVoice) {
                        ForEach(viewModel.availableVoices, id: \.self) { voice in
                            Text(voice).tag(voice)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .disabled(viewModel.availableVoices.isEmpty || viewModel.isGenerating)
                }
                Spacer()
                VStack(alignment: .leading) {
                    Text("Language")
                        .font(.subheadline)
                    Picker("Language", selection: $viewModel.language) {
                        ForEach(TTSService.Language.allCases) { lang in
                            Text(lang.displayName).tag(lang)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
            }

            Stepper(value: $viewModel.steps, in: 1...30) {
                Text("Steps: \(viewModel.steps)")
            }

            VStack(alignment: .leading) {
                Text(String(format: "Speed: %.2f", viewModel.speed))
                    .font(.subheadline)
                Slider(value: $viewModel.speed, in: 0.75...1.4, step: 0.01)
            }

            VStack(alignment: .leading) {
                Text(String(format: "Silence between chunks: %.2fs", viewModel.silenceSeconds))
                    .font(.subheadline)
                Slider(value: $viewModel.silenceSeconds, in: 0.0...0.6, step: 0.05)
            }

            VStack(alignment: .leading) {
                Text("Compute units")
                    .font(.subheadline)
                Picker("Compute units", selection: $viewModel.computeUnits) {
                    ForEach(TTSService.ComputeUnits.allCases) { unit in
                        Text(unit.displayName).tag(unit)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .disabled(viewModel.isGenerating || viewModel.isLoadingModels)
            }
        }
    }

    private var actionSection: some View {
        HStack(spacing: 12) {
            Button(action: { viewModel.generate() }) {
                HStack {
                    if viewModel.isGenerating {
                        ProgressView()
                    }
                    Text(viewModel.isGenerating ? "Generating…" : "Generate")
                }
            }
            .disabled(viewModel.isGenerating || viewModel.isLoadingModels || viewModel.availableVoices.isEmpty)
            .buttonStyle(.borderedProminent)

            Button(action: { viewModel.togglePlay() }) {
                Text(viewModel.isPlaying ? "Stop" : "Play")
            }
            .disabled(viewModel.audioURL == nil || viewModel.isGenerating)
            .buttonStyle(.bordered)
        }
    }

    private var metricsSection: some View {
        Group {
            VStack(alignment: .leading, spacing: 6) {
                if let metrics = viewModel.metrics {
                    Text(String(format: "Audio: %.2fs • Elapsed: %.2fs • RTF: %.2fx", metrics.audioSeconds, metrics.elapsedSeconds, metrics.rtf))
                        .font(.subheadline)
                    Text(String(format: "DP %.2fs • TE %.2fs • VE %.2fs • Voc %.2fs",
                                metrics.timing.durationPredictor,
                                metrics.timing.textEncoder,
                                metrics.timing.vectorEstimator,
                                metrics.timing.vocoder))
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    if let before = metrics.memoryBeforeMB, let after = metrics.memoryAfterMB {
                        Text(String(format: "Memory footprint: %.1f MB → %.1f MB", before, after))
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }

                if let loadSeconds = viewModel.modelLoadSeconds {
                    let reason = viewModel.modelLoadReason?.displayName ?? "Load"
                    let units = viewModel.modelLoadComputeUnits?.displayName ?? "Unknown"
                    Text(String(format: "Model load (%@, %@): %.2fs", reason, units, loadSeconds))
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    private var samplesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Sample prompts")
                .font(.subheadline)
            ForEach(viewModel.samples) { sample in
                Button(sample.title) {
                    viewModel.text = sample.text
                    viewModel.language = sample.language
                }
                .buttonStyle(.bordered)
            }
        }
    }
}
