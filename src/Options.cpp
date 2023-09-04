#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <errno.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include "Options.h"

///////////
// errno //
///////////
extern int errno;

////////////
// getopt //
////////////
extern int opterr;
extern int optopt;
extern int optind;
extern char* optarg;

using namespace PPCast;

static const char* HELP_COMMAND = "help";

std::vector<Option*> Options::options;

int Options::parseOptions(int argc, char *const *argv) {
    // getopt configuration
    opterr = 1; // Print error messages

    // Build GNU option structs
    std::sort(options.begin(), options.end(), [](const Option* a, const Option* b) {
        return std::less<std::string>()(std::string(a->name()), std::string(b->name()));
    });
    std::vector<struct option> optstructs;
    optstructs.reserve(options.size() + 1);
    for (const Option* opt : options) {
        optstructs.push_back({
            opt->name().c_str(),
            opt->hasArg(),
            nullptr,
            0
        });
    }
    optstructs.push_back({HELP_COMMAND, 0, 0, 0});
    optstructs.push_back({0, 0, 0, 0});

    // Parse options
    int optionIndex = 0;
    int error = 0;
    while ((error = getopt_long(argc, argv, "", &optstructs[0], &optionIndex)) != -1) {
        if (error) {
            std::cerr << "Error while parsing options" << std::endl;
            return error;
        }

        const struct option& opt = optstructs[optionIndex];
        if (std::string(opt.name) == std::string(HELP_COMMAND)) {
            printHelp(std::cout, argv[0]);
            return 1;
        }

        assert(opt.has_arg);
        if ((error = options[optionIndex]->parse(optarg))) {
            return error;
        }
    }

    return 0;
}

void Options::printConfig(std::ostream &os) {
    // Compute padding
    uint32_t maxwidth = static_cast<uint32_t>((*std::max_element(
        options.begin(),
        options.end(),
        [](Option* a, Option* b) { return a->name().size() < b->name().size();}
    ))->name().size());

    // Print config
    std::cout << "Running with configuration: " << std::boolalpha << std::endl;
    for (const Option* opt : options)
        os << "\t" << std::setw(maxwidth) << opt->name() << " = " << *opt << std::endl;
    std::cout << std::noboolalpha;
}

void Options::printHelp(std::ostream &os, char *commandName) {
    // Compute padding
    uint32_t maxwidth = static_cast<uint32_t>((*std::max_element(
        options.begin(),
        options.end(),
        [](Option* a, Option* b) { return a->name().size() < b->name().size();}
    ))->name().size());

    // Print config
    std::cout << "Usage: " << commandName << " [--options]" << std::endl;
    std::cout << "Options:" << std::boolalpha << std::endl;
    for (const Option* opt : options)
        os << "\t" << std::setw(maxwidth) << opt->name() << ": " << opt->desc() << std::endl;
    std::cout << std::noboolalpha;
}

Option::Option(const std::string& name, const std::string& desc, int hasArg)
    : m_name(name)
    , m_desc(desc)
    , m_hasArg(hasArg)
{
    Options::options.push_back(this);
};

int BoolOption::parse(const char* arg) {
    if (arg == nullptr) {
        m_data = true;
        return 0;
    } else {
        std::string str(arg);
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        std::istringstream is(str);
        is >> std::boolalpha >> m_data;
        if (!is.good()) {
            std::cerr << "Error while parsing " << m_name << ": '" << arg << "' is not a valid boolean" << std::endl;
            return 1;
        }
        return 0;
    }
}

int UIntOption::parse(const char* arg) {
    const long parsed = strtol(arg, nullptr, 10);
    m_data = static_cast<uint32_t>(parsed);

    if (errno == ERANGE || parsed > UINT32_MAX) {
        std::cerr << "Warning: " << m_name << " exceeds representable values -- capping value at " << UINT32_MAX << std::endl;
        m_data = UINT32_MAX;
    }

    if (parsed > 0 || (arg[0] == '0' && arg[1] == '\0')) return 0;

    std::cerr << "Error while parsing " << m_name << ": '" << arg << "' is not a valid unsigned integer" << std::endl;
    return 1;
}

int FloatOption::parse(const char* arg) {
    const float parsed = strtof(arg, nullptr);
    m_data = static_cast<float>(parsed);

    if (errno == ERANGE) {
        std::cerr << "Warning: " << m_name << " exceeds representable values -- capping value at " << HUGE_VALF << std::endl;
    }

    if (parsed || arg[0] == '0') return 0;

    std::cerr << "Error while parsing " << m_name << ": '" << arg << "' is not a valid float" << std::endl;
    return 1;
}

int StringOption::parse(const char* arg) {
    m_data = std::string(arg);
    return 0;
}