// Linkage-only htsim Logged registry for the standalone study adapter.

#include "config.h"
#include "loggertypes.h"

#include <fstream>

LoggedManager::LoggedManager() = default;

void LoggedManager::add_logged(Logged* logged) {
    _idmap.push_back(logged);
}

void LoggedManager::dump_idmap() {
    std::ofstream output("idmap.txt");
    for (const Logged* logged : _idmap) {
        output << logged->get_id() << ' ' << logged->_name << '\n';
    }
}

LoggedManager Logged::_logged_manager;
Logged::id_t Logged::LASTIDNUM = 1;
