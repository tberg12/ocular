from __future__ import print_function
import re

dir = "src/main/java/edu/berkeley/cs/nlp/ocular/main/"

option_line_re   = re.compile(r"\s*@Option\(gloss = \"(.*)\"\)\s*")
opt_decl_line_re = re.compile(r"\s*public static \S+ (\S+) = (.*);\s*(// (.+))?\s*")

def println(s): print(s.strip()+"\n")




for fn in ["TrainLanguageModel", "InitializeFont", "TranscribeOrTrainFont"]:

	with open(dir+fn+".java") as f:
		println("\n\n### "+fn)

		lines = f.readlines()
		for (line_num, line) in enumerate(lines):
			line = line.replace("Relevant to line extraction.","")
			line = line.strip()
			if "public static void main(String[] args)" in line:
				break

			if line.startswith("// "):
				header = line[3:]
				println("\n"+header)

			elif line.startswith("@Option"):
				opt_line_match = option_line_re.match(line)
				gloss = opt_line_match.group(1).strip().replace("\\\"","\"")
				
				decl_line_match = opt_decl_line_re.match(lines[line_num+1])
				name = decl_line_match.group(1).strip()
				default = decl_line_match.group(2).strip()
				default_explanation = decl_line_match.group(4)
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



				println("* `-%s`:\n%s\n%s%s" % (name, gloss.strip(), default.strip(), " (%s)" % (default_explanation) if default_explanation else ""))

