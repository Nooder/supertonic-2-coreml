//
//  supertonic2_coreml_ios_testApp.swift
//  supertonic2-coreml-ios-test
//
//  Created by Nader Beyzaei on 2026-01-16.
//

import SwiftUI
import SwiftData

@main
struct supertonic2_coreml_ios_testApp: App {
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Item.self,
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
