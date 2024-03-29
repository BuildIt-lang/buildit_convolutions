BASE_DIR=$(shell pwd)
SRC_DIR=$(BASE_DIR)/src
BUILD_DIR?=$(BASE_DIR)/build
BUILDIT_DIR?=$(BASE_DIR)/buildit
INCLUDE_DIR=$(BASE_DIR)/include
RUNTIME_DIR=$(BASE_DIR)/runtime

SAMPLES_DIR=$(BASE_DIR)/samples

INCLUDES=$(wildcard $(INCLUDE_DIR)/*.h) $(wildcard $(INCLUDE_DIR)/*/*.h) $(wildcard $(BUILDIT_DIR)/include/*.h) $(wildcard $(BUILDIT_DIR)/include/*/*.h) $(BUILD_DIR)/gen_headers/gen/compiler_headers.h

INCLUDE_FLAG=-I$(INCLUDE_DIR) -I$(BUILDIT_DIR)/include -I$(BUILD_DIR)/gen_headers

SRCS=$(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*/*.cpp)
SAMPLES_SRCS=$(wildcard $(SAMPLES_DIR)/*.cpp)
OBJS=$(subst $(SRC_DIR),$(BUILD_DIR),$(SRCS:.cpp=.o))
SAMPLES=$(subst $(SAMPLES_DIR),$(BUILD_DIR),$(SAMPLES_SRCS:.cpp=))

$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(BUILD_DIR)/samples)
$(shell mkdir -p $(BUILD_DIR)/conv_functions)
$(shell mkdir -p $(BUILD_DIR)/pipeline)
$(shell mkdir -p $(BUILD_DIR)/gen_headers)
$(shell mkdir -p $(BUILD_DIR)/gen_headers/gen)
$(shell mkdir -p $(BASE_DIR)/scratch)

BUILDIT_LIBRARY_NAME=buildit
BUILDIT_LIBRARY_PATH=$(BUILDIT_DIR)/build

LIBRARY_NAME=conv
DEBUG ?= 0
ifeq ($(DEBUG),1)
CFLAGS=-g -std=c++11 -O0
LINKER_FLAGS=-rdynamic  -g -L$(BUILDIT_LIBRARY_PATH) -L$(BUILD_DIR) -l$(LIBRARY_NAME) -l$(BUILDIT_LIBRARY_NAME) -ldl
else
CFLAGS=-std=c++11 -O3
LINKER_FLAGS=-rdynamic  -L$(BUILDIT_LIBRARY_PATH) -L$(BUILD_DIR) -l$(LIBRARY_NAME) -l$(BUILDIT_LIBRARY_NAME) -ldl
endif

LIBRARY=$(BUILD_DIR)/lib$(LIBRARY_NAME).a

CFLAGS+=-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wmissing-declarations -Woverloaded-virtual -pedantic-errors -Wno-deprecated -Wdelete-non-virtual-dtor -Werror
# -fno-move-loop-invariants
all: executables 

.PHONY: subsystem
subsystem:
	make -C $(BUILDIT_DIR)

.PRECIOUS: $(BUILD_DIR)/samples/%.o
.PRECIOUS: $(BUILD_DIR)/conv_functions/%.o
.PRECIOUS: $(BUILD_DIR)/pipeline/%.o

$(BUILD_DIR)/gen_headers/gen/compiler_headers.h:
	echo "#pragma once" > $@
	echo "#define GEN_TEMPLATE_NAME \"$(BASE_DIR)/scratch/code_XXXXXX\"" >> $@
	echo "#define COMPILER_PATH \"$(CXX)\"" >> $@
	echo "#define INCLUDES \"$(RUNTIME_DIR)\"" >> $@

$(BUILD_DIR)/samples/%.o: $(SAMPLES_DIR)/%.cpp $(INCLUDES)
	$(CXX) $(CFLAGS) $< -o $@ $(INCLUDE_FLAG) -c 


.PHONY: $(BUILDIT_LIBRARY_PATH)/lib$(BUILDIT_LIBRARY_NAME).a
$(BUILD_DIR)/sample%: $(BUILD_DIR)/samples/sample%.o $(LIBRARY) $(BUILDIT_LIBRARY_PATH)/lib$(BUILDIT_LIBRARY_NAME).a subsystem
	$(CXX) -o $@ $< $(LINKER_FLAGS)

.PHONY: executables
executables: $(SAMPLES)

$(LIBRARY): $(OBJS)
	ar rv $(LIBRARY) $(OBJS)	

$(BUILD_DIR)/conv_functions/%.o: $(SRC_DIR)/conv_functions/%.cpp $(INCLUDES)
	$(CXX) $(CFLAGS) $< -o $@ $(INCLUDE_FLAG) -c

$(BUILD_DIR)/pipeline/%.o: $(SRC_DIR)/pipeline/%.cpp $(INCLUDES)
	$(CXX) $(CFLAGS) $< -o $@ $(INCLUDE_FLAG) -c

run: executables
	./build/sample1

clean:
	rm -rf $(BUILD_DIR)
