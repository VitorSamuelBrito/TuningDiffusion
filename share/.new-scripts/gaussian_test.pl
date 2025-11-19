#!/usr/bin/perl

## Made to generate gaussian distributions to be tested.

## Creating the vector
## step is 1/$count
## vector length is $count*100
my $count = 100000;
my $final = ($count * 100) + 1;
print "$final \n";

## creating the vector
my $filename = "vector.dat";
open(VECTORFILE,">$filename") or die "Can\'t open $filename\n";
for ( my $i = 1; $i < $final ; $i++)
{
	my $res = $i/$count;
	printf VECTORFILE ("%.18e\n", $res); ## %.18e
}
close(VECTORFILE) or warn $!; 

## commenting old function
#foreach $number (1, 2, 5, 17, 70) {
#	my $filename = "vector_D_$number.dat";
#	open(VECTORFILE,">$filename") or die "Can\'t open $filename\n";
#	for ( my $i = 1; $i < $final ; $i++)
#	{
#		my $res = $i/$count;
#		printf VECTORFILE ("%.18e\n", $res); ## %.18e
#	}
#	close(VECTORFILE) or warn $!; 
#}


## Reading the vector
##my $number = 1;
foreach $number (1, 2, 5, 17, 70) {
	my $filename = "vector.dat";
	open(VECTORFILE,"<$filename") or die "Can\'t open $filename\n";
	@vector_data=<VECTORFILE>;
	close(VECTORFILE) or warn $!; 


	##  testing if the vector was read.
	foreach $line (@vector_data) {
	#	print "Value of line is $line";
	#	printf ("The value is %.5f\n", $line); ## %.18e
	}


	$last = 0;
	$dt = 0;
	## Quick test: printing (actually now saving) the array 'sequence'
	## to be used in all tests
	open(INITIALFILE,">gaussian_D_$number\_dt.dat") or die "can\'t open new file\n";
	foreach $line (@vector_data) {
	#   print "value of a: $a\n";
		$dt = $line - $last;
		$last = $line;
	#	print "Now dt is $dt \n";
		my $result = gaussian($number, $dt);
		printf INITIALFILE ("%.18e  %.18e\n", $line, $result); ## %.18e
	}
	close(INITIALFILE) or  warn $!;

	print "We are using D as $number\n";

	## Quick test: printing (actually now saving) the array 'sequence'
	## to be used in all tests
	open(INITIALFILE,">gaussian_D_$number.dat") or die "can\'t open new file\n";
	foreach $line (@vector_data) {
	#   print "value of a: $a\n";
		my $result = gaussian($number, $line);
		printf INITIALFILE ("%.18e  %.18e\n", $line, $result); ## %.18e
	}
	close(INITIALFILE) or  warn $!;
}

sub gaussian {
	my($DIFF,$dt) = @_;
	# sd is the rms value of the distribution.
	$sd = 2*$DIFF*$dt;
	$sd=sqrt($sd);
	my($RR);
	until($RR){
  		$M1=rand();
		$M2=rand();
		$M1=2*($M1-0.5);
		$M2=2*($M2-0.5);
        $tmp1 = $M1*$M1 + $M2*$M2;
        if( $tmp1 <= 1.0 && $tmp1 >= 0.0 ){
	       	$tmp2 = $sd*sqrt( -2.0*log($tmp1)/$tmp1 );
        	$RR =  $M1*$tmp2;
			return $RR;
			}
  		}
	}
	
