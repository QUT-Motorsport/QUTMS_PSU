%%
close all; clear; clc; format shortG;
% Ensure this script is started before the AMS is powered on, as it must be 
% ready to respond to the AMS's charging check request
fprintf("AMS Monitor Starting\n")
ports = serialportlist("available");
aports = [];
DEFAULT_BAUD = 115200;
% Dealing with dev/cu.* on MacOS &(linux?)
if ismac || isunix % Confirm with a linux fuckboy
	for i = 1:length(ports)
		if startsWith(ports(i), "/dev/tty")
			aports = [aports; ports(i)];
		end
	end
else
	% Windows display all the ports
	aports = (ports);
end
fprintf("====================\n")
for i = 1:length(aports)
	fprintf("(%d) - %s\n", i, aports(i));
end
port = input("Please select a serial port: ");
s = serialport(aports(port), DEFAULT_BAUD);
fprintf("Connected to serial port, awaiting AMS wakeup sequence...\n");
fprintf("Press Stop Loop button on figure to terminate\n");
% writematrix(["Runtime" "Voltage" "Temperature"],"AMS_Logs/log.csv");
configureTerminator(s,"CR/LF");
%%
iter = 0;
hold on;
subplot(2, 1, 1);

title("BMS Voltages")
xlabel("Time (Seconds)");
ylabel("Voltage (Volts)");

lines1 = [line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0)];
xlim([0, inf]);

subplot(2, 1, 2);

title("BMS Temperatures")
xlabel("Time (Seconds)");
ylabel("Temperature (Degrees)");

lines2 = [line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0) line(iter, 0)];
xlim([0, inf]);

ButtonHandle = uicontrol('Style', 'PushButton', ...
                         'String', 'Stop and Save Log', ...
                         'Position',[10 10 125 30],...
                         'Callback', 'delete(gcbf),');

drawnow;

jlfid = fopen('jsonLog.json', 'a');
fprintf(jlfid, '{\n');
tlfid = fopen('textLog.txt', 'a');
                     
% I feel so dirty doing this, time to wash the guilt of the following code away
% with a beer
pause on;
while true
    pause(0.1);
	if ~ishandle(ButtonHandle)
        
		disp('Exiting...');
        fprintf(jlfid, '}');
        fclose(jlfid);
        fclose(tlfid);
        clear s;
		break;
    end
    while s.NumBytesAvailable > 0
        % The following scope is such dodgy, unscalable code. Do not do
        % this. I just cbf making a more elegant solution.
        data = readline(s);
        jsonAttempt = 0;
        didParse = false;
        try
            jsonAttempt = jsondecode(data);
            didParse = true;
        catch ME
            jsonAttempt = data;
            didParse = false;
        end
        
        if ~didParse
            % This is a regular log string, or our charge request
            if strncmpi(data, string(0x69FF69FE), strlength(data))
            % We got AMS charge check
                fprintf(tlfid, data+'\n');
                write(s, string(0x69006901), "uint8");
                %write(s, '1778346494\r\n', "uint8");
            else
                fprintf(tlfid, data);
                disp(data)
            end
        else
            fprintf(jlfid, data+',\n');
            % Yay! we have some json, time to parse
            if isfield(jsonAttempt, 'BalanceInfo')
                fprintf("Balance State for BMS-%v: %v @ %v Volts\r\n", jsonAttempt.BalanceInfo.BMS + 1, fliplr(de2bi(jsonAttempt.BalanceInfo.BalanceState), jsonAttempt.BalanceInfo.BalanceVoltage));
                %TODO
            elseif isfield(jsonAttempt, 'VoltageInfo')
                lines1(jsonAttempt.VoltageInfo.BMS + 1).YData = [lines1(jsonAttempt.VoltageInfo.BMS + 1).YData mean(jsonAttempt.VoltageInfo.Voltages)];
                lines1(jsonAttempt.VoltageInfo.BMS + 1).XData = [lines1(jsonAttempt.VoltageInfo.BMS + 1).XData jsonAttempt.VoltageInfo.RT];
            elseif isfield(jsonAttempt, 'TemperatureInfo')
                lines2(jsonAttempt.TemperatureInfo.BMS + 1).YData = [lines2(jsonAttempt.TemperatureInfo.BMS + 1).YData mean(jsonAttempt.TemperatureInfo.Temperatures)];
                lines2(jsonAttempt.TemperatureInfo.BMS + 1).XData = [lines2(jsonAttempt.TemperatureInfo.BMS + 1).XData jsonAttempt.TemperatureInfo.RT];
            end
        end
        
        % if our data is x long, compare it to our expected startup string
        % sequence which by the way was chosen carefully and totally not
        % because i am a child
%         if strncmpi(data, string(0x69FF69FE), strlength(data))
%             % We got AMS charge check
%             write(s, string(0x69006901), "uint8");
%         elseif length(V) == 12 % Voltages
%             st = struct;
%             st.runtime = V(1);
%             st.bmsId = V(2) + 1;
%             st.voltages = V(3:12);
% 
%             lines1(st.bmsId).YData = [lines1(st.bmsId).YData mean(st.voltages)];
%             lines1(st.bmsId).XData = [lines1(st.bmsId).XData st.runtime/1000];
%         elseif length(V) == 14 % Temperatures
%             st = struct;
%             st.runtime = V(1);
%             st.bmsId = V(2) + 1; % OHHHH LOOK AT ME IM MATLAB AND I MAKE ARRAYS START AT ONE. CRINGE.
%             st.temperatures = V(3:14);
%             
%             lines2(st.bmsId).YData = [lines2(st.bmsId).YData mean(st.temperatures)];
%             lines2(st.bmsId).XData = [lines2(st.bmsId).XData st.runtime/1000];
%         elseif length(V) == 4
%             % Do nothing, its not a value we want to see
%         else
%             disp(data);
%         end
    end
end