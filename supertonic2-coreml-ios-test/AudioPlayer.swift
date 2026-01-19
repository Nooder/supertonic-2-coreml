//
//  AudioPlayer.swift
//  supertonic2-coreml-ios-test
//
//  Created by Codex.
//

import Foundation
import AVFoundation

final class AudioPlayer: NSObject, AVAudioPlayerDelegate {
    private var player: AVAudioPlayer?
    private var onFinish: (() -> Void)?

    func play(url: URL, onFinish: (() -> Void)? = nil) {
        self.onFinish = onFinish
        do {
            let session = AVAudioSession.sharedInstance()
            // Keep playback audible while allowing other audio to keep playing.
            try session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
            try session.setActive(true)
            let data = try Data(contentsOf: url)
            let player = try AVAudioPlayer(data: data)
            player.delegate = self
            player.volume = 1.0
            player.prepareToPlay()
            player.play()
            self.player = player
        } catch {
            print("Audio play error: \(error)")
            onFinish?()
        }
    }

    func stop() {
        player?.stop()
        player = nil
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onFinish?()
    }
}
