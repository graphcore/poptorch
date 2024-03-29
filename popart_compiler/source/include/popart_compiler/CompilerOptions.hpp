// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>

#include <popart/commgroup.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/popx/devicexmanager.hpp>

#include "popart_compiler/PopartEnums.hpp"

namespace poptorch {
namespace popart_compiler {
namespace detail {

enum class ExecutionMode { Pipelined, Sharded, Phased, N };

// To be kept in sync with the Liveness python enum in python/enums.py
enum class Liveness {
  AlwaysLive,
  OffChipAfterFwd,
  OffChipAfterFwdNoOverlap,
  OffChipAfterEachPhase,
  N
};

struct CompilerOptions {
  // A constant to tell the copmiler to use the system ipu version
  constexpr static std::uint64_t use_system_ipu_version =
      std::numeric_limits<std::uint64_t>::max();

  // Make PopART save the initializers in a separate file.
  // (Needed to keep the ONNX protobuf below the 2GB limit when compiling
  // large models)
  std::string external_initializers_file;
  // Number of times the graph will be executed for each execution.
  std::uint64_t steps{0};
  // Strategy to adopt for returning the graph's output tensors.
  PopartOutputMode output_mode;
  // 'N' when output_mode == PopartOutputMode::EveryN
  std::uint64_t output_return_period;
  // True if running on the model, False otherwise.
  bool ipu_model{false};
  // Automatically round up the number of IPUs, if required, to the minimum
  // number required to be reserved
  bool auto_round_num_ipus{false};
  // Only used for offline compilation (DeviceConnectionType.Never): version
  // of the IPU should the Poplar compiler be targeting.
  std::uint64_t ipu_version{use_system_ipu_version};
  // ID of the specific IPU the user wants to use. (If not set we'll just
  // iterate over the IPUs present on the system and try to connect to one
  // that matches our requirements).
  std::uint64_t ipu_id{0};
  popart::DeviceConnectionType connection_type;
  popart::SyncPattern sync_pattern;
  std::uint64_t random_seed{0};

  // The frontend will unpack the user option and pass it directly in as
  // [IPU_ID] = Memory proportion for that IPU
  std::unordered_map<std::uint32_t, float> available_memory_proportion;

  // When running in distributed mode: number of processes the training is
  // split// over.
  std::uint64_t num_distributed_processes{1};
  // In distributed mode: unique ID of this process in [0,
  // num_distributed_processes]// range
  std::uint64_t distributed_process_id{0};

  popart::Patterns patterns{popart::PatternsLevel::Default};
  ExecutionMode execution_mode{};

  // Phased execution options: see the python documentation for more
  // information about how to use them
  //
  // Here is how they translate into Popart options:
  // serial_phases_execution: True -> executionPhaseSettings.stages = 1
  //                          False-> executionPhaseSettings.stages = 2
  //
  // separate_backward_phase:
  //  False:
  //   fwd:       bwd:
  //   phase 0 -> phase 4
  //   phase 1 -> phase 3
  //   phase 2 -> phase 2
  //
  // (End of fwd and start of bwd are part of the same phase)
  //  True:
  //   fwd:       bwd:
  //   phase 0 -> phase 6
  //   phase 1 -> phase 5
  //   phase 2 -> phase 4
  //
  //  This is done by setting options.executionPhaseSettings.phases to N+1
  //
  //  Note that the bwd phases begin with phase 4 and not phase 3. This is
  //  because PopART requires the phase IDs of a fwd/bwd pair to have matching
  //  parity. Since the fwd phase ID is 2, the next phase ID with even parity
  //  is 4.
  //
  //  Furthermore, all odd phases must run on the same IPUs, and all even
  //  phases must also run on the same IPUs.
  //
  // tensors_liveness:
  //  Note: tensors have a liveness of [phase, phase+2]
  //  AlwaysLive:
  //   fwd:       bwd:
  //   phase 0 -> phase 6
  //   phase 1 -> phase 5
  //   phase 2 -> phase 4
  // Stride = 1
  //
  //  OffChipAfterFwd:
  //   fwd:       bwd:
  //   phase 0 -> phase 8
  //   phase 1 -> phase 7
  //   phase 2 -> phase 6
  // Stride = 1
  // (Gap between fwd and bwd > 2)
  //  This is done by incrementing options.executionPhaseSettings.phases by 3
  //
  //  OffChipAfterFwdNoOverlap:
  //   fwd:       bwd:
  //   phase 0 -> phase 12
  //   phase 2 -> phase 10
  //   phase 4 -> phase 8
  // Stride = 2
  // (Gap between fwd and bwd > 2, with no overlapping of load/store)
  //  This is done by incrementing options.executionPhaseSettings.phases by 3
  //  and multiplying the phase_id by 2.
  //
  //  OffChipAfterEachPhase: (Only for stage=1)
  //   fwd:       bwd:
  //   phase 0 -> phase 20
  //   phase 4 -> phase 16
  //   phase 8 -> phase 12
  // Stride = 4
  // (Gap between each phase > 2)
  // This is done by incrementing options.executionPhaseSettings.phases by 3
  // and multiplying the phase_id by 4.
  bool serial_phases_execution{false};
  bool separate_backward_phase{false};
  Liveness tensors_liveness{};

  // Debug name for the model
  std::string model_name;

  // (Not yet supported) Whether each buffer should be broadcasted from the
  // first to other replicas on each training step.
  bool broadcast_buffers{false};

  std::int64_t input_group_size;
  popart::CommGroupType input_cgt;
};

} // namespace detail
} // namespace popart_compiler
} // namespace poptorch
