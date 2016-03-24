package edu.berkeley.cs.nlp.ocular.main;

import java.text.SimpleDateFormat;
import java.util.Date;

import fig.OptionsParser;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
abstract class OcularRunnable implements Runnable {

	private SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
	
	final protected void doMain(OcularRunnable main, String[] args) {
		System.out.println(toArgListString(args));
		long startTime = System.currentTimeMillis();
		printStartTime(startTime);
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] { main });
		if (!parser.doParse(args)) System.exit(1);
		main.validateOptions();
		main.run();
		long endTime = System.currentTimeMillis();
		printEndTime(startTime, endTime);
	}

	abstract protected void validateOptions();

	private static String toArgListString(String[] args) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < args.length; ++i) {
		  if (i % 2 == 0) 
			  sb.append("  " + args[i]);
		  else 
			  sb.append(" " + args[i] + "\n");
		}
		if (sb.codePointAt(sb.length()-1) != '\n')
			sb.append("\n");
		return sb.toString();
	}
	
	private void printEndTime(long startTime, long endTime) {
		System.out.println("\n"+ convertSecondsToAmountOfTimeString(endTime - startTime) + " elapsed. Completed at "+sdf.format(new Date(endTime)));
	}

	private void printStartTime(long startTime) {
		System.out.println("Started job at "+sdf.format(new Date(startTime))+"\n");
	}

	private String convertSecondsToAmountOfTimeString(long millis) {
		long seconds = millis / 1000;
	    long s = seconds % 60;
	    long m = (seconds / 60) % 60;
	    long h = (seconds / (60 * 60));
	    return String.format("%02d:%02d:%02d", h,m,s);
	}

}
