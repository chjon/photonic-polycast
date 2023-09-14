#ifndef PPCAST_OPTIONS_H
#define PPCAST_OPTIONS_H

#include <getopt.h>
#include "Common.h"

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
        /// @brief A registry of all the program options
        static std::vector<Option*> options;
        friend class Option;

    public:
        /**
         * @brief Parse options from the command line
         * 
         * @param argc The number of program arguments
         * @param argv The list of program arguments
         * @return 0 if parsing was successful; otherwise, return an error code
         */
        static int parseOptions(int argc, char *const *argv);

        /**
         * @brief Print the program configuration based on the given command line options
         * 
         * @param os The output stream
         */
        static void printConfig(std::ostream &os);

        /**
         * @brief Print a help string for the program and descriptions of each option
         * 
         * @param os The output stream
         * @param commandName The name of the program
         */
        static void printHelp(std::ostream &os, char *commandName);
    };

    /**
     * @brief An abstract class representing a program option
     * 
     */
    class Option {
    protected:
        /// @brief The name of the option
        const std::string m_name;

        /// @brief A description of the opbtion
        const std::string m_desc;

        /// @brief Whether the option requires an argument (see getopt documentation for details)
        const int m_hasArg;

        /**
         * @brief Print the option as a string -- this is used as a virtual override of the
         * @code{ostream} object's @code{operator<<} method
         * 
         * @param os The output stream
         * @return The output stream -- returned for chaining the << operator
         */
        virtual std::ostream& toStream(std::ostream &os) const = 0;

    public:
        /**
         * @brief Construct a new Option object
         * 
         * @param name The name of the option
         * @param desc A description of the option
         * @param hasArg Whether the option requires an argument (see getopt documentation for details)
         */
        Option(const std::string& name, const std::string& desc, int hasArg);

        /**
         * @brief Get the name of the option
         * 
         * @return the name of the option
         */
        const std::string& name() const { return m_name; }

        /**
         * @brief Get a description of the option
         * 
         * @return a description of the option
         */
        const std::string& desc() const { return m_desc; }

        int hasArg() const { return m_hasArg; }
        virtual int parse(const char* arg) = 0;

    private:
        friend std::ostream& operator<<(std::ostream &os, const Option& opt) {
            return opt.toStream(os);
        }
    };

    class BoolOption : public Option {
    protected:
        bool m_data;

        virtual std::ostream& toStream(std::ostream &os) const override { return os << m_data; }

    public:
        BoolOption(const std::string& name, const std::string& desc, bool defaultVal)
            : Option(name, desc + " (default: " + (defaultVal ? "true" : "false") + ")", optional_argument)
            , m_data(defaultVal)
        {}

        bool operator* () const { return m_data; }
        virtual int parse(const char* arg) override;
    };

    class UIntOption : public Option {
    protected:
        uint32_t m_data;

        virtual std::ostream& toStream(std::ostream &os) const override { return os << m_data; }

    public:
        UIntOption(const std::string& name, const std::string& desc, uint32_t defaultVal)
            : Option(name, desc + " (default: " + std::to_string(defaultVal) + ")", required_argument)
            , m_data(defaultVal)
        {}

        uint32_t operator* () const { return m_data; }
        virtual int parse(const char* arg) override;
    };

    class FloatOption : public Option {
    protected:
        float m_data;

        virtual std::ostream& toStream(std::ostream &os) const override { return os << m_data; }

    public:
        FloatOption(const std::string& name, const std::string& desc, float defaultVal)
            : Option(name, desc + " (default: " + std::to_string(defaultVal) + ")", required_argument)
            , m_data(defaultVal)
        {}

        float operator* () const { return m_data; }
        virtual int parse(const char* arg) override;
    };

    class StringOption : public Option {
    protected:
        std::string m_data;

        virtual std::ostream& toStream(std::ostream &os) const override { return os << '"' << m_data << '"'; }

    public:
        StringOption(const std::string& name, const std::string& desc, const std::string& defaultVal)
            : Option(name, desc + " (default: \"" + defaultVal + "\")", required_argument)
            , m_data(defaultVal)
        {}

        const std::string& operator* () const { return m_data; }
        virtual int parse(const char* arg) override;
    };
}

#endif