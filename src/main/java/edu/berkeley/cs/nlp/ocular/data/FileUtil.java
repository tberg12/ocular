package edu.berkeley.cs.nlp.ocular.data;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class FileUtil {

	public static List<File> recursiveFiles(String name) {
		return FileUtil.recursiveFiles(name, null);
	}

	public static List<File> recursiveFiles(File file) {
		return FileUtil.recursiveFiles(file, null);
	}

	public static List<File> recursiveFiles(String name, Set<String> validExtensions) {
		return FileUtil.recursiveFiles(new File(name), validExtensions);
	}

	public static List<File> recursiveFiles(File file, Set<String> validExtensions) {
		List<File> files = new ArrayList<File>();
		if (file.isDirectory()) {
			for (File f : file.listFiles()) {
				if (!f.getName().startsWith(".")) { // ignore hidden files
					files.addAll(recursiveFiles(f, validExtensions));
				}
			}
		}
		else {
			if (validExtensions == null || validExtensions.contains(extension(file.getName()))) {
				files.add(file);
			}
		}
		return files;
	}

	public static List<File> recursiveFiles(List<String> names) {
		return FileUtil.recursiveFiles(names, null);
	}

	public static List<File> recursiveFiles(List<String> names, Set<String> validExtensions) {
		List<File> files = new ArrayList<File>();
		for (String f : names)
			files.addAll(FileUtil.recursiveFiles(f, validExtensions));
		return files;
	}

	public static String extension(String name) {
		String[] split = name.split("\\.");
		return split[split.length - 1];
	}

	public static String withoutExtension(String name) {
		return name.replaceAll("\\.[^.]*$", "");
	}

}
