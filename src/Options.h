#ifndef PPCAST_OPTIONS_H
#define PPCAST_OPTIONS_H

#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

//////////////////////////////
// Configuration for getopt //
//////////////////////////////
extern int opterr;
extern int optopt;
extern int optind;
extern char* optarg;

namespace PPCast {
    class Option;

    /**
     * @brief Utility class for parsing options
     * 
     */
    class Options {
    private:
        static std::vector<Option*> options;
        friend class Option;

    public:
        static int parseOptions(int argc, char *const *argv);
        static void printConfig(std::ostream &os);
        static void printHelp(std::ostream &os, char *commandName);
    };

    class Option {
    protected:
        const std::string m_name;
        const std::string m_desc;
        const int         m_hasArg;

        virtual std::ostream& toStream(std::ostream &os) const = 0;

    public:
        Option(const std::string& name, const std::string& desc, int hasArg);

        const std::string& name() const { return m_name; }
        const std::string& desc() const { return m_desc; }

        int hasArg() const { return m_hasArg; }
        virtual int parse(const char* arg) = 0;

    private:
        friend std::ostream& operator<<(std::ostream &os, const Option& opt) {
            return opt.toStream(os);
        }
    };

    class BoolOption : protected Option {
    protected:
        bool m_data;

        virtual std::ostream& toStream(std::ostream &os) const { return os << m_data; }

    public:
        BoolOption(const std::string& name, const std::string& desc, bool defaultVal)
            : Option(name, desc + " (default: " + (defaultVal ? "true" : "false") + ")", optional_argument)
            , m_data(defaultVal)
        {}

        bool operator* () const { return m_data; }
        virtual int parse(const char* arg);
    };

    class UIntOption : protected Option {
    protected:
        uint32_t m_data;

        virtual std::ostream& toStream(std::ostream &os) const { return os << m_data; }

    public:
        UIntOption(const std::string& name, const std::string& desc, uint32_t defaultVal)
            : Option(name, desc + " (default: " + std::to_string(defaultVal) + ")", required_argument)
            , m_data(defaultVal)
        {}

        uint32_t operator* () const { return m_data; }
        virtual int parse(const char* arg);
    };

    class FloatOption : protected Option {
    protected:
        float m_data;

        virtual std::ostream& toStream(std::ostream &os) const { return os << m_data; }

    public:
        FloatOption(const std::string& name, const std::string& desc, float defaultVal)
            : Option(name, desc + " (default: " + std::to_string(defaultVal) + ")", required_argument)
            , m_data(defaultVal)
        {}

        float operator* () const { return m_data; }
        virtual int parse(const char* arg);
    };

    class StringOption : protected Option {
    protected:
        std::string m_data;

        virtual std::ostream& toStream(std::ostream &os) const { return os << '"' << m_data << '"'; }

    public:
        StringOption(const std::string& name, const std::string& desc, const std::string& defaultVal)
            : Option(name, desc + " (default: \"" + defaultVal + "\")", required_argument)
            , m_data(defaultVal)
        {}

        const std::string& operator* () const { return m_data; }
        virtual int parse(const char* arg);
    };
}

#endif