from __future__ import print_function
from collections import defaultdict
import re


main_dir = "src/main/java/edu/berkeley/cs/nlp/ocular/main/"

option_line_re        = re.compile(r"\s*(//\s*)?@Option\(gloss = \"(.*)\"\)\s*")
gloss_re              = re.compile(r".*gloss = \"([^\"]+)\".*")
opt_decl_line_re      = re.compile(r"\s*(//\s*)?public static \S+ (\S+) = (.*);\s*(// (.+))?\s*")
default_quoted_int_re = re.compile(r'"\d+"')

# def println(s): print(s.strip()+"\n")

def first_line_that(lines, f):
    for i,line in enumerate(lines):
        if f(line): return (i,line)
    raise RuntimeError("no line found that satisfies condition")

def split_on_blanks(lines):
    buffer = []
    for l in list(lines)+['']:
        if not l.strip():
            if buffer:
                yield buffer
                buffer = []
        else:
            buffer.append(l)

def no_newlines(f):
    for l in f:
        if l[-1] == '\n':
            l = l[:-1]
        yield l


program_options = dict()
with open('options_lists.txt') as olf:
    for program_options_lines in split_on_blanks(no_newlines(olf)):
        program_options_list = []
        for line in program_options_lines[1:]:
            if line.startswith('\t'):
                program_options_list.append(line.strip())
        program_name = program_options_lines[0].replace('#','').strip()
        program_options[program_name] = program_options_list



main_class_names = ["InitializeLanguageModel", "InitializeFont", "TrainFont", "Transcribe"]
class_file_names = main_class_names+["FonttrainTranscribeShared", "LineExtractionOptions", "OcularRunnable"]
assert not ((set(program_options) | set(main_class_names)) - (set(program_options) & set(main_class_names))), '%s' % ((set(program_options) | set(main_class_names)) - (set(program_options) & set(main_class_names)))



# Read the option information from the source code files
class_to_options = dict() # dict[fn -> dict[opt_name -> (gloss, default, default_explanation)]]
for class_file_name in class_file_names:

    class_options = dict()
    with open(main_dir+class_file_name+".java") as f:
        lines = list(no_newlines(f))
        for (line_num, line) in enumerate(lines):
            line = line.strip()
            line = line.replace("\\\"","<QUOTE>") # hide escaped quotes
            if line.startswith("// "): continue
            if "public static void main(String[] args)" in line: break

            if "@Option" in line:
                gloss_match = gloss_re.match(line)
                if not gloss_match: print('gloss_match failed on [%s]' % (line))
                gloss = gloss_match.group(1)
                
                decl_line_match = opt_decl_line_re.match(lines[line_num+1])
                name = decl_line_match.group(2).strip()
                default = decl_line_match.group(3).strip()
                if default_quoted_int_re.match(default): default = default[1:-1]
                default_explanation = decl_line_match.group(5)
                if default_explanation: default_explanation = default_explanation.strip()

                if "Default:" in gloss:
                    split_gloss = re.match("(.*)(Default: .*)", gloss)
                    gloss = split_gloss.group(1)
                    default = split_gloss.group(2)
                    default_explanation = ""
                elif "Required" in gloss:
                    split_gloss = re.match("(.*)(Required .*)", gloss)
                    gloss = split_gloss.group(1)
                    default = split_gloss.group(2)
                    default_explanation = ""
                elif default_explanation and default_explanation.startswith("Default: "):
                    default = default_explanation
                    default_explanation = ""
                elif default == "null" and default_explanation:
                    # println(">>>> ["+default_explanation+"] "+str(default_explanation.startswith("Required")))
                    if default_explanation and default_explanation.startswith("Required"):
                        default = default_explanation
                        default_explanation = ""
                    elif default_explanation:
                        default = "Default: "+default_explanation
                        default_explanation = ""
                elif default == "Integer.MAX_VALUE" and default_explanation:
                    default = "Default: "+default_explanation
                    default_explanation = ""
                else:
                    default = "Default: "+default

                class_options[name] = (gloss.strip(), default.strip(), default_explanation)

    class_to_options[class_file_name] = class_options

# Get a list of the direct superclasses of each class
super_classes = dict()
for class_file_name in class_file_names:
    with open(main_dir+class_file_name+".java") as f:
        lines = list(no_newlines(f))
        _,class_decl_line = first_line_that(lines, lambda l: l.strip().startswith("public class") or l.strip().startswith("public abstract class"))
        super_classes[class_file_name] = set(class_decl_line.split()) - set((class_file_name+' public abstract class extends implements Runnable {').split())

# Add options from superclasses to the class options dictionaries
running_super_classes = dict((c,list(scs)) for (c,scs) in super_classes.iteritems())
while set(c for (c,scs) in running_super_classes.iteritems() if scs):
    for c,scs in running_super_classes.iteritems():
        for sc in scs:
            if not running_super_classes[sc]:
                assert not set(class_to_options[c]) & set(class_to_options[sc])
                class_to_options[c].update(class_to_options[sc])
                scs.remove(sc)

# for c in main_class_names:
#     print(c)
#     for o in sorted(class_to_options[c]):
#         print(o)
#     print()


# Check that the option_lists.txt file matches the actual source code
for (name, list_opts) in program_options.iteritems():
    source_options = class_to_options[name]
    assert set(list_opts) == set(source_options), "%s" % ((set(list_opts) | set(source_options)) - (set(list_opts) & set(source_options)))


# Print out the README content
program_options = dict()
with open('options_lists.txt') as olf:
    for program_options_lines in split_on_blanks(no_newlines(olf)):
        program_name = program_options_lines[0].replace('#','').strip()
        for line in program_options_lines:
            if line.startswith('\t'):
                name = line.strip()
                (gloss, default, default_explanation) = class_to_options[program_name][name]
                print("* `-%s`:" % (name))
                print("%s" % (gloss.replace("<QUOTE>",'"').strip()))
                print("%s%s" % (default.strip(), " (%s)" % (default_explanation.replace("<QUOTE>",'"')) if default_explanation else ""))
            else:
                print(line)
            print()


# Check that option descriptions are consistent
option_all_descriptions = defaultdict(list)
for (c,os) in class_to_options.iteritems():
	for (o,d) in os.iteritems():
		option_all_descriptions[o].append((d,c))
for (o,ds) in option_all_descriptions.iteritems():
	if len(set(d for (d,c) in ds)) > 1:
		print("MULTIPLE DESCRIPTIONS FOUND: %s: %s" % (o, '\n   '.join(['']+sorted(map(str,ds)))))

























