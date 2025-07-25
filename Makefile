# Compiler and flags
CXX := clang++
CXXFLAGS := -std=c++17 -mavx2 -mfma `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

# Directories
SRCDIR := srcs
INCDIR := -Iincludes -Ilib/Tensorium_lib/includes
OBJDIR := .objs

# Target binary
TARGET := JubretonAnalyser

# Source and object files
SRCFILES := $(wildcard $(SRCDIR)/*.cpp)
OBJFILES := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCFILES))

# Default target
all: $(TARGET)

# Link object files into final binary
$(TARGET): $(OBJFILES)
	@echo "Linking $(TARGET)..."
	$(CXX) $(OBJFILES) -o $(TARGET) $(LDFLAGS)

# Compile .cpp to .o
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@

# Create object directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean build files
clean:
	rm -rf $(OBJDIR)

# Clean everything including binary
fclean: clean
	rm -f $(TARGET)

# Rebuild from scratch
re: fclean all

.PHONY: all clean fclean re

