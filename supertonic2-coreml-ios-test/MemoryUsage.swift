//
//  MemoryUsage.swift
//  supertonic2-coreml-ios-test
//
//  Created by Codex.
//

import Foundation
import Darwin.Mach

struct MemoryUsage {
    static func currentFootprintMB() -> Double? {
        // Uses mach task info to approximate real app memory footprint (in MB).
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
        let kerr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundedPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), reboundedPtr, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return nil }
        return Double(info.phys_footprint) / 1024.0 / 1024.0
    }
}
