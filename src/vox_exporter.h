#pragma once

#include <vector>

class avatar;
class creature_tracker;
class map;
class npc;

namespace vox_exporter
{

void start( int port = 9876 );
void stop();
void on_turn_end( const avatar &you, const map &here, const creature_tracker &creatures,
                  const std::vector<const npc *> &npcs );

} // namespace vox_exporter
