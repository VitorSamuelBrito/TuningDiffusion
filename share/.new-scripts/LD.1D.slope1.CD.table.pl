#!/usr/bin/perl
$X=<STDIN>;
chomp($X);
$DIFFX=<STDIN>; ##  um2/s
chomp($DIFFX);

$STEPS=<STDIN>;
chomp($STEPS);
$dt=<STDIN>; ## ns
chomp($dt);

$basin1=<STDIN>; 
chomp($basin1);
$basin2=<STDIN>; 
chomp($basin2);
$HEIGHT=<STDIN>;
chomp($HEIGHT);
$SLOPE=<STDIN>;
chomp($SLOPE);


$M=($basin2+$basin1)/2;
$D=($basin2-$basin1)/2;


$NG=<STDIN>;
chomp($NG);

for($I=0;$I<$NG;$I++){
	$MAX[$I]=<STDIN>;
	chomp($MAX[$I]);
	$sigma[$I]=<STDIN>;
	chomp($sigma[$I]);
	$GH[$I]=<STDIN>;
	chomp($GH[$I]);
}

open(file,">TRAJECTORY") or die "can\'t open trajectory\n";
open(fileS,">SURFACE") or die "can\'t open trajectory\n";

$I=0;
$bounds=100;
$grids=100000;
$width=$bounds*1.000000/$grids;

while($I<$grids){
	$I++;
	$X=$I*$width;

##	$RRR=gaussian($DIFF,$dt);
##	print "$RRR\n";

	$FX=0;
	$EE=0;

	$grad=grad24($M,$D,$HEIGHT,$X);
	$FX+=$grad;

	$grad=E24($M,$D,$HEIGHT,$X);
	$EE+=$grad;

	for($J=0;$J<$NG;$J++){

	$grad=gradG($MAX[$J],$sigma[$J],$GH[$J],$X);
	$FX+=$grad;
	$grad=EG($MAX[$J],$sigma[$J],$GH[$J],$X);
	$EE+=$grad;

	}
        $EE+=$SLOPE*$X;
	$FX+=$SLOPE;
	$FF[$I]=$FX;
	printf fileS ("%10.6f %10.6f %10.6f\n", $X, $EE, $FX);

}



$I=0;
while($I<$STEPS){
        $I++;

##      $RRR=gaussian($DIFF,$dt);
##      print "$RRR\n";

	$J=$X/$width;
        $FX=$FF[$J];

        $X+=-$DIFFX*$dt*$FX+gaussian($DIFFX,$dt);

        if($I % 100==0){
        $T=$dt*$I;
        printf file ("%5.2f %5.2f\n", $T, $X);
        }
}


sub grad24 {
	my($M,$D,$HEIGHT,$X) = @_;
	$grad=(-2*$HEIGHT*2*($X-$M)/$D**2 +4*$HEIGHT*($X-$M)**3/$D**4);
	return $grad
}

sub E24 {
	my($M,$D,$HEIGHT,$X) = @_;
	$E=(-$HEIGHT*2*($X-$M)**2/$D**2 +$HEIGHT*($X-$M)**4/$D**4);
	return $E
}

sub gradG {
        my($MAX,$sigma,$HEIGHT,$X) = @_;
	$grad=$HEIGHT*exp(-($X-$MAX)**2/$sigma**2)*2*($MAX-$X)/$sigma**2;
        return $grad
}

sub EG {
        my($MAX,$sigma,$HEIGHT,$X) = @_;
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
