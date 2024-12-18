#!/usr/bin/perl

## Functions extracted from original files.
## Data generated with this script will be used to test the python funcitons


## Sequence created without zero to avoid division by zero errors
#@positive = (1 .. 50);
#@negative = (-50 .. -1);
#my @sequence = (@negative, @positive);
my @unsorted_sequence;

## Number of samples
my $samples = 20;

## Creating the array sequence to be used in all tests
for (my $i = 1; $i <= $samples; $i++){
	$random = rand(100) - 50;
	push(@unsorted_sequence, $random);
}

# Sorting numbers with use of 
# spaceship operator 
@sequence = sort { $a <=> $b } @unsorted_sequence; 


## Quick test: printing the array 'unsorted_sequence'
#foreach $a (@unsorted_sequence) {
#	print "value of a: $a\n";
#}

## Quick test: printing (actually now saving) the array 'sequence'
## to be used in all tests
open(initialfile,">sequence.dat") or die "can\'t open new file\n";
foreach $a (@sequence) {
#   print "value of a: $a\n";
	printf initialfile ("%10.6f\n", $a);
}


## Quick test: printing the results for a function
#foreach $a (@sequence) {
#	$result = DDsin($a, 1, 1, 1);	
#	printf ("value of a and result : %5.2f %5.2f\n", $a, $result);
#}



## Creating the files
open(file,">test.dat") or die "can\'t open new file\n";
foreach $a (@sequence){
	foreach $b (@sequence){
		foreach $c (@sequence){
			foreach $d (@sequence){
				$result = DDsin($a, $b, $c, $d);
				printf file ("%10.6f %10.6f %10.6f %10.6f %10.6f\n", $a, $b, $c, $d, $result);
			}
		}
	}
}



sub DDsin {
	my($DIFFX, $SINM, $X, $SINF) = @_;
	$DD=$DIFFX+$SINM*exp(-$X/$SINF);
	return $DD;

}


sub DDsinslope {
        my($DIFFX, $SINM, $X, $SINF) = @_;
        $DDslope=(-$SINM/$SINF)*exp(-$X/$SINF);
        return $DDslope;

}



sub DDexp {
	my($DIFFX, $SINM, $X, $SINF) = @_;
	$DD=$DIFFX+$SINM*exp(-$X/$SINF);
	return $DD;

}


sub DDexpslope {
        my($DIFFX, $SINM, $X, $SINF) = @_;
        $DDslope=(-$SINM/$SINF)*exp(-$X/$SINF);
        return $DDslope;

}


sub grad24 {
        my($M, $D, $HEIGHT, $X) = @_;
        $grad=(-2*$HEIGHT*2*($X-$M)/$D**2 +4*$HEIGHT*($X-$M)**3/$D**4);
        return $grad
}


sub E24 {
        my($M, $D, $HEIGHT, $X) = @_;
        $E=(-$HEIGHT*2*($X-$M)**2/$D**2 +$HEIGHT*($X-$M)**4/$D**4);
        return $E
}


sub gradG {
        my($MAX, $sigma, $HEIGHT, $X) = @_;
        $grad=$HEIGHT*exp(-($X-$MAX)**2/$sigma**2)*2*($MAX-$X)/$sigma**2;
        return $grad
}


sub EG {
        my($MAX, $sigma, $HEIGHT, $X) = @_;
        $E=$HEIGHT*exp(-($X-$MAX)**2/$sigma**2);
        return $E
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

