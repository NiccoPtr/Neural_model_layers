### Model editing



* Using (hebb - xor) rule, allows for D2 layers to learn in a more proper manner since their low str activity threshold for learning allows learning in cases of high dopamine (which causes low D2 activity)



* Need for PFCd\_PPC to drive attention and lead to input presence in the environment based on ATTENTION:



##### &#x09;PROBLEMS



&#x09;	- If PFCd\_PPC activity leads to input presence, you need to figure out a way such that the input present is not based on the PFCd\_PPC activity of trial before



* ATTENTIONAL DRIVE would avoid unmatching conditions since the input will be based on PFCd\_PPC activity, so it would be matching. Plus, MC would allow dopamine release for learning only if it matches, thus decreased uncongruent conditions.







* Need to check for learning hebb - xor in BGv since the input is onset, thus the "- xor" component creates damages
* Also, need to check for lasting duration of DA signal given by the VTA, it shouldn't last longer than the BLA input otherwise wrong learnings will happen

&#x09;

&#x09;- Need to work on Matrices LH\_VTA \& Food\_LH power



* Solve the double spikes in the VTA through the LH



&#x09;- Need to allow first input to let the LH spike (conditional), but the second input (unconditional) shouldn't permit a second spike



* Need for less lock in activity and more random activity to learn



&#x09;- Check for theta\_DA for learning, baseline DA



\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_



Changes to be made to the behavioural model during the learning phase:



* Set a baseline DA value other than zero (e.g. 0.2) and a maximum achievable value for tonic DA caused by lock-in in the striatal ‘disinhibition’ components (e.g. 0.5)



* Make the phasic peak of DA as rapid and instantaneous as possible so as not to affect subsequent timesteps



* Add the presence of one of the two manipulande via the scheduling function:



&#x09;- If attention is directed towards a manipulanda that is not present in the environment (not initialised by scheduling), the input will not allow the manipulanda to be entered



&#x09;- inp \[n] = 1.0 if attention\[n] and state\[n], else inp\[0:2] = 0.0



\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_



Need to increase the amount of spontaneous actions at low DA levels while maintaining CS bias and devaluation bias:



* Increase random activity within the MC and PFCd\_PPC, trying both noise ratio or DA sensitivity



* Permit the influence from the NAc to guide the action choice in case of Devaluation test



* Enhance the influence of NAc and DMS onto DA release from the SNpc



* Pay attention to theta\_DA for striatal learning, probably it will have to be increased

