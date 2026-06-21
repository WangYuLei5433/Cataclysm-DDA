#include "vox_exporter.h"

#if defined(CATACLYSM_VOX_EXPORT)

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include "avatar.h"
#include "creature_tracker.h"
#include "field.h"
#include "map.h"
#include "map_scale_constants.h"
#include "monster.h"
#include "mtype.h"
#include "npc.h"
#include "trap.h"
#include "veh_type.h"
#include "vehicle.h"
#include "vpart_position.h"
#include "weather.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <locale>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Forward-declare ungetch to avoid pulling in curses macros (which conflict
// with CDDA symbols such as weather_type::clear).
extern "C" int ungetch( int ch );

namespace
{

// ─── Z-level scan range ───────────────────────────────────────────────────────
// Scan this many layers below and above the player's current Z level.
static constexpr int Z_SCAN_BELOW = 2;
static constexpr int Z_SCAN_ABOVE = 2;
static constexpr int Z_SCAN_LAYERS = Z_SCAN_BELOW + 1 + Z_SCAN_ABOVE; // 5

// ─── Tile snapshot for delta detection ───────────────────────────────────────
struct tile_snap {
    std::string ter;
    std::string furn;            // "" = no furniture
    std::string trap;            // "" = no trap (tr_null)
    std::string field;           // "" = no field (dominant type)
    int         field_intensity = 0;
    bool        has_items = false;
    std::string vehicle;         // "" = no vehicle, else displayed vpart id

    bool operator==( const tile_snap &o ) const {
        return ter == o.ter && furn == o.furn && trap == o.trap &&
               field == o.field && field_intensity == o.field_intensity &&
               has_items == o.has_items && vehicle == o.vehicle;
    }
    bool operator!=( const tile_snap &o ) const { return !( *this == o ); }
};

// Cache covers MAPSIZE_X * MAPSIZE_Y * Z_SCAN_LAYERS entries.
// Z index: z_off = z - (player_z - Z_SCAN_BELOW), range [0, Z_SCAN_LAYERS).
// The player_z shifts when the player changes Z → detect via prev_player_z.
// The XY bubble shifts when abs_sub XY changes → detect via prev_abs_sub_xy.
static constexpr size_t TILE_CACHE_SIZE =
    static_cast<size_t>( MAPSIZE_X ) * MAPSIZE_Y * Z_SCAN_LAYERS;

std::vector<tile_snap> tile_cache( TILE_CACHE_SIZE );
bool cache_initialized = false;

// Track previous scan origin to detect when cache is no longer valid.
int prev_player_z   = 0;
int prev_abs_sub_x  = 0;
int prev_abs_sub_y  = 0;

inline size_t cache_idx( int x, int y, int z_off )
{
    return static_cast<size_t>( z_off ) * MAPSIZE_X * MAPSIZE_Y +
           static_cast<size_t>( y ) * MAPSIZE_X +
           static_cast<size_t>( x );
}

// ─── WebSocket / server state ─────────────────────────────────────────────────
std::atomic<bool> running{ false };
std::atomic<int>  turn_counter{ 0 };
std::atomic<bool> full_scan_pending{ true };
std::thread       server_thread;
std::mutex        socket_mutex;
std::mutex        frame_mutex;
std::mutex        keys_mutex;
std::queue<int>   pending_keys;
int client_socket  = -1;
int listen_socket  = -1;
bool has_game_frame       = false;
std::string latest_full_frame; // always a frame_full — sent to new clients on connect

// ─── JSON helpers ─────────────────────────────────────────────────────────────
void set_json_locale( std::ostringstream &out )
{
    out.imbue( std::locale::classic() );
}

std::string json_escape( const std::string &input )
{
    std::ostringstream out;
    for( const char ch : input ) {
        switch( ch ) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b";  break;
            case '\f': out << "\\f";  break;
            case '\n': out << "\\n";  break;
            case '\r': out << "\\r";  break;
            case '\t': out << "\\t";  break;
            default:   out << ch;     break;
        }
    }
    return out.str();
}

std::string json_time()
{
    const int tick = to_turns<int>( calendar::turn - calendar::turn_zero );
    std::ostringstream out;
    set_json_locale( out );
    out << "\"time\":{\"turn\":" << tick
        << ",\"hour\":"   << hour_of_day<int>( calendar::turn )
        << ",\"minute\":" << minute_of_hour<int>( calendar::turn )
        << ",\"is_night\":" << ( is_night( calendar::turn ) ? "true" : "false" ) << "}";
    return out.str();
}

std::string json_weather()
{
    return "\"weather\":\"" + json_escape( get_weather().weather_id.str() ) + "\"";
}

// ─── Tile snapshot helper ─────────────────────────────────────────────────────
tile_snap snap_tile( const map &here, const tripoint_bub_ms &p )
{
    tile_snap s;
    s.ter = here.ter( p ).id().str();
    s.furn = here.has_furn( p ) ? here.furn( p ).id().str() : "";

    const trap &tr = here.tr_at( p );
    // trap::operator!= compares trap::loadid against the integer trap_id
    s.trap = ( tr != tr_null ) ? tr.id.str() : "";

    const field &f = here.field_at( p );
    for( const std::pair<const field_type_id, field_entry> &kv : f ) {
        // Pick first (dominant) field entry.
        s.field           = kv.first.id().str();
        s.field_intensity = kv.second.get_field_intensity();
        break;
    }

    s.has_items = here.has_items( p );
    const optional_vpart_position veh = here.veh_at( p );
    if( veh.has_value() ) {
        const auto disp = veh->part_displayed();
        s.vehicle = disp.has_value() ? disp->info().id.str() : "vp_frame";
    }
    return s;
}

// ─── Tile JSON serialisation ──────────────────────────────────────────────────
void append_tile_json( std::ostringstream &out, int x, int y, int z,
                       const tile_snap &s, const map &here )
{
    const tripoint_bub_ms p( x, y, z );
    const float light_raw = here.ambient_light_at( p );
    const double light    = std::clamp( static_cast<double>( light_raw ) / 60.0, 0.15, 1.0 );

    out << "{\"x\":" << x << ",\"y\":" << y << ",\"z\":" << z
        << ",\"ter\":\""  << json_escape( s.ter ) << "\"";

    if( !s.furn.empty() ) {
        out << ",\"furn\":\"" << json_escape( s.furn ) << "\"";
    }
    if( !s.trap.empty() ) {
        out << ",\"trap\":\"" << json_escape( s.trap ) << "\"";
    }
    if( !s.field.empty() ) {
        out << ",\"field\":\"" << json_escape( s.field )
            << "\",\"field_intensity\":" << s.field_intensity;
    }
    if( s.has_items ) {
        out << ",\"has_items\":true";
    }
    if( !s.vehicle.empty() ) {
        out << ",\"vehicle\":\"" << json_escape( s.vehicle ) << "\"";
    }
    out << ",\"light\":" << std::fixed << std::setprecision( 2 ) << light << "}";
}

// ─── Creature JSON ────────────────────────────────────────────────────────────
static const char *facing_str( FacingDirection f )
{
    return f == FacingDirection::LEFT ? "L" : "R";
}

void append_creatures_json( std::ostringstream &out,
                            int z_min, int z_max,
                            const creature_tracker &creatures,
                            const std::vector<const npc *> &npcs )
{
    out << "\"creatures\":[";
    bool first = true;

    for( const shared_ptr_fast<monster> &mon : creatures.get_monsters_list() ) {
        if( !mon || mon->is_dead() ) {
            continue;
        }
        const tripoint_bub_ms mp = mon->pos_bub();
        if( mp.z() < z_min || mp.z() > z_max ) {
            continue;
        }

        if( !first ) { out << ","; }
        first = false;

        out << "{\"id\":"    << creatures.temporary_id( *mon )
            << ",\"type\":\"" << json_escape( mon->type->id.str() ) << "\""
            << ",\"x\":"     << mp.x()
            << ",\"y\":"     << mp.y()
            << ",\"z\":"     << mp.z()
            << ",\"hp_ratio\":" << std::fixed << std::setprecision( 2 )
            << std::clamp( mon->hp_percentage() / 100.0, 0.0, 1.0 )
            << ",\"facing\":\"" << facing_str( mon->facing ) << "\"}";
    }

    for( const npc *guy : npcs ) {
        if( !guy || guy->is_dead_state() ) {
            continue;
        }
        const tripoint_bub_ms np = guy->pos_bub();
        if( np.z() < z_min || np.z() > z_max ) {
            continue;
        }

        if( !first ) { out << ","; }
        first = false;

        const int id = guy->getID().is_valid() ? guy->getID().get_value() : -1;
        out << "{\"id\":\"npc_" << id << "\""
            << ",\"type\":\"npc_survivor\""
            << ",\"x\":"     << np.x()
            << ",\"y\":"     << np.y()
            << ",\"z\":"     << np.z()
            << ",\"hp_ratio\":" << std::fixed << std::setprecision( 2 )
            << std::clamp( guy->hp_percentage() / 100.0, 0.0, 1.0 )
            << ",\"facing\":\"" << facing_str( guy->facing ) << "\"}";
    }

    out << "]";
}

// ─── Player JSON ──────────────────────────────────────────────────────────────
void append_player_json( std::ostringstream &out, const avatar &you, int z_min, int z_max )
{
    const tripoint_bub_ms pos = you.pos_bub();
    const int stamina_max     = std::max( 1, you.get_stamina_max() );
    const int stamina_pct     = std::clamp( you.get_stamina() * 100 / stamina_max, 0, 100 );

    out << "\"player\":{"
        << "\"x\":"        << pos.x()
        << ",\"y\":"       << pos.y()
        << ",\"z\":"       << pos.z()
        << ",\"facing\":\"" << facing_str( you.facing ) << "\""
        << ",\"hp\":"      << std::clamp( you.hp_percentage(), 0, 100 )
        << ",\"stamina\":" << stamina_pct
        << "},"
        << "\"viewport\":{"
        << "\"center\":[" << pos.x() << "," << pos.y() << "," << pos.z() << "]"
        << ",\"z_min\":"  << z_min
        << ",\"z_max\":"  << z_max
        << "}";
}

// ─── Full frame (entire Reality Bubble, current Z±N layers) ──────────────────
std::string build_full_frame( const avatar &you, const map &here,
                              const creature_tracker &creatures,
                              const std::vector<const npc *> &npcs )
{
    const int tick       = turn_counter.load();
    const int player_z   = you.pos_bub().z();
    const int z_min      = std::max( player_z - Z_SCAN_BELOW, -OVERMAP_DEPTH );
    const int z_max      = std::min( player_z + Z_SCAN_ABOVE,  OVERMAP_HEIGHT );

    std::ostringstream out;
    set_json_locale( out );
    out << "{\"type\":\"frame_full\",\"tick\":" << tick << ",";
    append_player_json( out, you, z_min, z_max );
    out << ",\"tiles\":[";

    bool first_tile = true;
    for( int z = z_min; z <= z_max; ++z ) {
        const int z_off = z - ( player_z - Z_SCAN_BELOW );
        for( int y = 0; y < MAPSIZE_Y; ++y ) {
            for( int x = 0; x < MAPSIZE_X; ++x ) {
                const tripoint_bub_ms p( x, y, z );
                if( !here.inbounds( p ) ) {
                    continue;
                }
                const tile_snap s = snap_tile( here, p );

                // Update cache.
                tile_cache[cache_idx( x, y, z_off )] = s;

                if( !first_tile ) { out << ","; }
                first_tile = false;
                append_tile_json( out, x, y, z, s, here );
            }
        }
    }

    out << "],";
    append_creatures_json( out, z_min, z_max, creatures, npcs );
    out << "," << json_time() << "," << json_weather() << "}";
    return out.str();
}

// ─── Delta frame (only changed tiles since last full/delta) ──────────────────
std::string build_delta_frame( const avatar &you, const map &here,
                               const creature_tracker &creatures,
                               const std::vector<const npc *> &npcs )
{
    const int tick       = turn_counter.load();
    const int player_z   = you.pos_bub().z();
    const int z_min      = std::max( player_z - Z_SCAN_BELOW, -OVERMAP_DEPTH );
    const int z_max      = std::min( player_z + Z_SCAN_ABOVE,  OVERMAP_HEIGHT );

    std::ostringstream out;
    set_json_locale( out );
    out << "{\"type\":\"frame_delta\",\"tick\":" << tick << ",";
    append_player_json( out, you, z_min, z_max );
    out << ",\"tiles\":[";

    bool first_tile = true;
    for( int z = z_min; z <= z_max; ++z ) {
        const int z_off = z - ( player_z - Z_SCAN_BELOW );
        for( int y = 0; y < MAPSIZE_Y; ++y ) {
            for( int x = 0; x < MAPSIZE_X; ++x ) {
                const tripoint_bub_ms p( x, y, z );
                if( !here.inbounds( p ) ) {
                    continue;
                }
                const tile_snap s = snap_tile( here, p );
                tile_snap &cached = tile_cache[cache_idx( x, y, z_off )];
                if( s == cached ) {
                    continue; // no change, skip
                }
                cached = s;

                if( !first_tile ) { out << ","; }
                first_tile = false;
                append_tile_json( out, x, y, z, s, here );
            }
        }
    }

    out << "],";
    append_creatures_json( out, z_min, z_max, creatures, npcs );
    out << "," << json_time() << "," << json_weather() << "}";
    return out.str();
}

// ─── WebSocket framing ────────────────────────────────────────────────────────
std::string base64_encode( const unsigned char *data, size_t len )
{
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve( ( ( len + 2 ) / 3 ) * 4 );

    for( size_t i = 0; i < len; i += 3 ) {
        const uint32_t octet_a = data[i];
        const uint32_t octet_b = i + 1 < len ? data[i + 1] : 0;
        const uint32_t octet_c = i + 2 < len ? data[i + 2] : 0;
        const uint32_t triple  = ( octet_a << 16 ) | ( octet_b << 8 ) | octet_c;

        out.push_back( alphabet[( triple >> 18 ) & 0x3f] );
        out.push_back( alphabet[( triple >> 12 ) & 0x3f] );
        out.push_back( i + 1 < len ? alphabet[( triple >> 6 ) & 0x3f] : '=' );
        out.push_back( i + 2 < len ? alphabet[triple & 0x3f] : '=' );
    }
    return out;
}

uint32_t left_rotate( uint32_t value, uint32_t count )
{
    return ( value << count ) | ( value >> ( 32 - count ) );
}

std::array<unsigned char, 20> sha1( const std::string &input )
{
    std::vector<unsigned char> bytes( input.begin(), input.end() );
    const uint64_t bit_len = static_cast<uint64_t>( bytes.size() ) * 8;
    bytes.push_back( 0x80 );
    while( bytes.size() % 64 != 56 ) {
        bytes.push_back( 0 );
    }
    for( int i = 7; i >= 0; --i ) {
        bytes.push_back( static_cast<unsigned char>( bit_len >> ( i * 8 ) ) );
    }

    uint32_t h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe;
    uint32_t h3 = 0x10325476, h4 = 0xc3d2e1f0;

    for( size_t chunk = 0; chunk < bytes.size(); chunk += 64 ) {
        uint32_t w[80] = {};
        for( int i = 0; i < 16; ++i ) {
            const size_t j = chunk + i * 4;
            w[i] = ( bytes[j] << 24 ) | ( bytes[j + 1] << 16 ) |
                   ( bytes[j + 2] << 8 ) | bytes[j + 3];
        }
        for( int i = 16; i < 80; ++i ) {
            w[i] = left_rotate( w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1 );
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for( int i = 0; i < 80; ++i ) {
            uint32_t f = 0, k = 0;
            if( i < 20 )      { f = ( b & c ) | ( ( ~b ) & d ); k = 0x5a827999; }
            else if( i < 40 ) { f = b ^ c ^ d;                   k = 0x6ed9eba1; }
            else if( i < 60 ) { f = ( b & c ) | ( b & d ) | ( c & d ); k = 0x8f1bbcdc; }
            else               { f = b ^ c ^ d;                   k = 0xca62c1d6; }
            const uint32_t temp = left_rotate( a, 5 ) + f + e + k + w[i];
            e = d; d = c; c = left_rotate( b, 30 ); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    std::array<unsigned char, 20> digest = {};
    const uint32_t words[5] = { h0, h1, h2, h3, h4 };
    for( int i = 0; i < 5; ++i ) {
        digest[i * 4]     = static_cast<unsigned char>( words[i] >> 24 );
        digest[i * 4 + 1] = static_cast<unsigned char>( words[i] >> 16 );
        digest[i * 4 + 2] = static_cast<unsigned char>( words[i] >> 8 );
        digest[i * 4 + 3] = static_cast<unsigned char>( words[i] );
    }
    return digest;
}

std::string websocket_accept_key( const std::string &client_key )
{
    const std::string magic = client_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    const auto digest = sha1( magic );
    return base64_encode( digest.data(), digest.size() );
}

std::string header_value( const std::string &request, const std::string &name )
{
    const std::string needle = name + ":";
    size_t pos = request.find( needle );
    if( pos == std::string::npos ) {
        return "";
    }
    pos += needle.size();
    while( pos < request.size() && ( request[pos] == ' ' || request[pos] == '\t' ) ) {
        ++pos;
    }
    const size_t end = request.find( "\r\n", pos );
    return request.substr( pos, end == std::string::npos ? std::string::npos : end - pos );
}

bool send_all( int fd, const unsigned char *data, size_t len )
{
    size_t sent = 0;
    while( sent < len ) {
        const ssize_t result = send( fd, data + sent, len - sent, MSG_NOSIGNAL );
        if( result <= 0 ) {
            return false;
        }
        sent += static_cast<size_t>( result );
    }
    return true;
}

bool send_text_frame( int fd, const std::string &payload )
{
    std::vector<unsigned char> frame;
    frame.push_back( 0x81 );
    if( payload.size() < 126 ) {
        frame.push_back( static_cast<unsigned char>( payload.size() ) );
    } else if( payload.size() <= 0xffff ) {
        frame.push_back( 126 );
        frame.push_back( static_cast<unsigned char>( payload.size() >> 8 ) );
        frame.push_back( static_cast<unsigned char>( payload.size() ) );
    } else {
        frame.push_back( 127 );
        for( int i = 7; i >= 0; --i ) {
            frame.push_back( static_cast<unsigned char>( payload.size() >> ( i * 8 ) ) );
        }
    }
    frame.insert( frame.end(), payload.begin(), payload.end() );
    return send_all( fd, frame.data(), frame.size() );
}

void close_client()
{
    std::lock_guard<std::mutex> lock( socket_mutex );
    if( client_socket >= 0 ) {
        close( client_socket );
        client_socket = -1;
    }
}

std::string current_full_frame()
{
    std::lock_guard<std::mutex> lock( frame_mutex );
    if( has_game_frame ) {
        return latest_full_frame;
    }
    return "{\"type\":\"waiting\",\"reason\":\"no_game_frame\"}";
}

void broadcast( const std::string &payload )
{
    std::lock_guard<std::mutex> lock( socket_mutex );
    if( client_socket >= 0 && !send_text_frame( client_socket, payload ) ) {
        close( client_socket );
        client_socket = -1;
    }
}

void send_current_full_frame( int fd )
{
    const std::string frame = current_full_frame();
    std::lock_guard<std::mutex> lock( socket_mutex );
    if( client_socket == fd && !send_text_frame( fd, frame ) ) {
        close( fd );
        client_socket = -1;
    }
}

void handle_client_payload( int fd, const std::string &payload )
{
    if( payload.find( "\"request_full\"" ) != std::string::npos ) {
        send_current_full_frame( fd );
        return;
    }
    if( payload.find( "\"input_key\"" ) != std::string::npos ) {
        const size_t pos = payload.find( "\"key\":" );
        if( pos != std::string::npos ) {
            size_t start = pos + 6;
            while( start < payload.size() &&
                   ( payload[start] == ' ' || payload[start] == '"' ) ) {
                ++start;
            }
            if( start < payload.size() ) {
                std::lock_guard<std::mutex> lock( keys_mutex );
                pending_keys.push( static_cast<int>( payload[start] ) );
            }
        }
    }
}

void poll_client_messages()
{
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock( socket_mutex );
        fd = client_socket;
    }
    if( fd < 0 ) {
        return;
    }

    unsigned char buffer[4096] = {};
    const ssize_t received = recv( fd, buffer, sizeof( buffer ), MSG_DONTWAIT );
    if( received == 0 ) {
        close_client();
        return;
    }
    if( received < 0 ) {
        if( errno != EAGAIN && errno != EWOULDBLOCK ) {
            close_client();
        }
        return;
    }

    size_t offset = 0;
    while( offset + 2 <= static_cast<size_t>( received ) ) {
        const unsigned char first  = buffer[offset++];
        const unsigned char second = buffer[offset++];
        const unsigned char opcode = first & 0x0f;
        const bool masked          = ( second & 0x80 ) != 0;
        uint64_t payload_len       = second & 0x7f;

        if( payload_len == 126 ) {
            if( offset + 2 > static_cast<size_t>( received ) ) { return; }
            payload_len = ( static_cast<uint64_t>( buffer[offset] ) << 8 ) | buffer[offset + 1];
            offset += 2;
        } else if( payload_len == 127 ) {
            if( offset + 8 > static_cast<size_t>( received ) ) { return; }
            payload_len = 0;
            for( int i = 0; i < 8; ++i ) {
                payload_len = ( payload_len << 8 ) | buffer[offset + i];
            }
            offset += 8;
        }

        unsigned char mask[4] = {};
        if( masked ) {
            if( offset + 4 > static_cast<size_t>( received ) ) { return; }
            std::memcpy( mask, buffer + offset, sizeof( mask ) );
            offset += 4;
        }

        if( payload_len > static_cast<uint64_t>( received ) - offset ) { return; }

        if( opcode == 0x8 ) {
            close_client();
            return;
        }

        if( opcode == 0x1 ) {
            std::string payload;
            payload.reserve( static_cast<size_t>( payload_len ) );
            for( uint64_t i = 0; i < payload_len; ++i ) {
                unsigned char byte = buffer[offset + i];
                if( masked ) { byte ^= mask[i % 4]; }
                payload.push_back( static_cast<char>( byte ) );
            }
            handle_client_payload( fd, payload );
        }

        offset += static_cast<size_t>( payload_len );
    }
}

bool perform_handshake( int fd )
{
    std::string request;
    char buffer[1024];
    while( request.find( "\r\n\r\n" ) == std::string::npos ) {
        const ssize_t n = recv( fd, buffer, sizeof( buffer ), 0 );
        if( n <= 0 ) { return false; }
        request.append( buffer, buffer + n );
        if( request.size() > 8192 ) { return false; }
    }

    const std::string key = header_value( request, "Sec-WebSocket-Key" );
    if( key.empty() ) { return false; }

    const std::string response =
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " + websocket_accept_key( key ) + "\r\n\r\n";
    return send_all( fd, reinterpret_cast<const unsigned char *>( response.data() ), response.size() );
}

void handle_client( int fd )
{
    // Keep socket blocking so send_all() reliably delivers large full frames.
    // poll_client_messages() already uses MSG_DONTWAIT for non-blocking reads.
    {
        std::lock_guard<std::mutex> lock( socket_mutex );
        if( client_socket >= 0 ) {
            close( client_socket );
        }
        client_socket = fd;
    }

    // New client: force a full rescan on the next turn so it gets a complete map.
    full_scan_pending.store( true );

    send_text_frame( fd, "{\"type\":\"hello\",\"version\":\"0.3.0\","
                     "\"cdda_version\":\"vox-m3\","
                     "\"world_name\":\"CDDA Live Exporter\","
                     "\"tile_size\":1.0,"
                     "\"z_scan_below\":" + std::to_string( Z_SCAN_BELOW ) + ","
                     "\"z_scan_above\":" + std::to_string( Z_SCAN_ABOVE ) + "}" );
    send_text_frame( fd, current_full_frame() );
}

void server_loop( int port )
{
    listen_socket = socket( AF_INET, SOCK_STREAM, 0 );
    if( listen_socket < 0 ) { return; }

    int opt = 1;
    setsockopt( listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof( opt ) );

    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl( INADDR_ANY );
    addr.sin_port        = htons( static_cast<uint16_t>( port ) );

    if( bind( listen_socket, reinterpret_cast<sockaddr *>( &addr ), sizeof( addr ) ) != 0 ||
        listen( listen_socket, 4 ) != 0 ) {
        close( listen_socket );
        listen_socket = -1;
        return;
    }

    fcntl( listen_socket, F_SETFL, fcntl( listen_socket, F_GETFL, 0 ) | O_NONBLOCK );

    while( running.load() ) {
        sockaddr_in client_addr = {};
        socklen_t len = sizeof( client_addr );
        const int fd  = accept( listen_socket, reinterpret_cast<sockaddr *>( &client_addr ), &len );
        if( fd >= 0 ) {
            if( perform_handshake( fd ) ) {
                handle_client( fd );
            } else {
                close( fd );
            }
        }

        poll_client_messages();
        std::this_thread::sleep_for( std::chrono::milliseconds( 25 ) );
    }

    close_client();
    if( listen_socket >= 0 ) {
        close( listen_socket );
        listen_socket = -1;
    }
}

} // anonymous namespace

// ─── Public API ───────────────────────────────────────────────────────────────
namespace vox_exporter
{

void start( int port )
{
    if( running.exchange( true ) ) { return; }
    full_scan_pending.store( true );
    cache_initialized = false;
    server_thread = std::thread( server_loop, port );
}

void stop()
{
    if( !running.exchange( false ) ) { return; }
    if( server_thread.joinable() ) {
        server_thread.join();
    }
}

void on_turn_end( const avatar &you, const map &here,
                  const creature_tracker &creatures,
                  const std::vector<const npc *> &npcs )
{
    turn_counter.fetch_add( 1 );

    // Detect horizontal bubble shift (player crossed a submap boundary).
    // When this happens the local (x,y) coordinates no longer map to the same
    // absolute tiles, so the cache is stale and we need a full rescan.
    const tripoint_abs_sm abs_sub = here.get_abs_sub();
    if( !cache_initialized ||
        abs_sub.x() != prev_abs_sub_x ||
        abs_sub.y() != prev_abs_sub_y ||
        you.pos_bub().z() != prev_player_z ) {
        full_scan_pending.store( true );
        prev_abs_sub_x  = abs_sub.x();
        prev_abs_sub_y  = abs_sub.y();
        prev_player_z   = you.pos_bub().z();
        cache_initialized = true;
    }

    const bool do_full = full_scan_pending.exchange( false );
    const std::string frame = do_full
                              ? build_full_frame( you, here, creatures, npcs )
                              : build_delta_frame( you, here, creatures, npcs );

    {
        std::lock_guard<std::mutex> lock( frame_mutex );
        // Only cache frame_full frames so new clients get a coherent snapshot.
        if( do_full ) {
            latest_full_frame = frame;
            has_game_frame    = true;
        }
    }
    broadcast( frame );

    // Inject any pending key inputs from the browser into the ncurses input queue.
    std::lock_guard<std::mutex> lock( keys_mutex );
    while( !pending_keys.empty() ) {
        ungetch( pending_keys.front() );
        pending_keys.pop();
    }
}

} // namespace vox_exporter

#else

namespace vox_exporter
{

void start( int ) {}
void stop() {}
void on_turn_end( const avatar &, const map &, const creature_tracker &,
                  const std::vector<const npc *> & ) {}

} // namespace vox_exporter

#endif
